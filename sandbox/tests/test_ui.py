"""Browser smoke test of the app (sandbox/static/): the real server in a thread, a headless Chromium through
Playwright. Skipped when Playwright or the server cannot be imported, or when no Chromium is installed."""
from __future__ import annotations

import glob
import json
import os
import threading
from pathlib import Path

import pytest

sync_api = pytest.importorskip("playwright.sync_api")
server_mod = pytest.importorskip("sandbox.server")


@pytest.fixture
def base_url(repo_root, tmp_path):
    """The sandbox served on an ephemeral port, workspace and saved studies in tmp_path (nothing is written to
    the repo), for the length of one test."""
    server = server_mod.make_server(repo_root, host="127.0.0.1", port=0, workspace=tmp_path / "ws")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _launch(p):
    """Chromium as Playwright installed it; failing that, any Chromium already under PLAYWRIGHT_BROWSERS_PATH
    (a Playwright upgrade can leave the browser build it expects missing); else skip."""
    try:
        return p.chromium.launch()
    except Exception as first:
        root = os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or str(Path.home() / ".cache" / "ms-playwright")
        candidates = sorted(glob.glob(os.path.join(root, "chromium-*", "chrome-linux*", "chrome")), reverse=True)
        candidates += sorted(glob.glob(os.path.join(root, "chromium_headless_shell-*", "chrome-linux*", "*headless_shell")), reverse=True)
        for exe in candidates:
            try:
                return p.chromium.launch(executable_path=exe)
            except Exception:
                continue
        pytest.skip(f"no usable Chromium for Playwright: {first}")


def _validate_repo(page, base_url) -> dict:
    """POST /api/study/validate of the repo preset, as the app sends it."""
    spec = page.request.get(base_url + "/api/studies/repo").json()["spec"]
    res = page.request.post(base_url + "/api/study/validate", data=json.dumps({"spec": spec}),
                            headers={"Content-Type": "application/json", "X-Sandbox": "1"})
    assert res.ok, res.text()
    return res.json()


def test_app_smoke(base_url):
    errors: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = _launch(p)
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            page.goto(base_url + "/#/study")

            # The study picker lists the repo study, and it is the one loaded.
            page.wait_for_selector("select.study-picker option[value='repo']", state="attached", timeout=15000)
            assert page.eval_on_selector("select.study-picker", "e => e.value") == "repo"
            assert "LLM political polarization" in page.input_value("input[data-path='name']")

            # It validates without errors (warnings are allowed), and is the exact repo study: the badge says
            # "exact" only on the server's repo_exact (grid, catalogue and instrument equal the repo's files),
            # never on repo_shaped alone; a server that does not report repo_exact gets the neutral engine badge.
            page.wait_for_selector(".status-bar .pill-ok, .status-bar .pill-warn, .status-bar .pill-error", timeout=15000)
            pill = page.inner_text(".status-bar .pill")
            assert "error" not in pill, pill
            validation = _validate_repo(page, base_url)
            assert validation["repo_shaped"] is True, validation.get("repo_shape_reason")
            badge = page.locator(".status-bar [data-repo]")
            assert badge.count() == 1
            if "repo_exact" in validation:
                assert validation["repo_exact"] is True, validation.get("repo_exact_reason")
                assert badge.get_attribute("data-repo") == "exact"
                assert badge.inner_text() == "exact repo study"
                assert page.locator(".status-bar .badge-repo").count() == 1
            else:
                assert badge.get_attribute("data-repo") == "shaped"
                assert "exact" not in badge.inner_text()
            page.wait_for_selector(".view-study .stats .stat", timeout=10000)
            assert page.locator(".view-study table tbody tr").count() >= 2

            # The persona preview renders a treated cell of the repo study (placeholder catalogue text).
            page.goto(base_url + "/#/personas")
            page.wait_for_function(
                "() => { const e = document.querySelector('[data-testid=persona-text]'); return e && e.textContent.includes('Placeholder'); }",
                timeout=15000,
            )
            assert page.locator(".preview-out .error-box").count() == 0

            # The other design views render.
            for view in ("axes", "instrument", "models", "run"):
                page.goto(base_url + f"/#/{view}")
                page.wait_for_selector(f".view-{view} .view-header h1", timeout=10000)
                assert page.locator(f".view-{view} .error-box").count() == 0, view

            # The run list renders (an empty data dir is fine).
            page.goto(base_url + "/#/runs")
            page.wait_for_selector(".view-runs .view-header h1", timeout=10000)
            page.wait_for_function(
                "() => { const v = document.querySelector('.view-runs'); return v && !v.textContent.includes('Loading'); }",
                timeout=15000,
            )
            assert page.locator(".view-runs .error-box").count() == 0

            page.goto(base_url + "/#/jobs")
            page.wait_for_selector(".view-jobs .view-header h1", timeout=10000)
            page.wait_for_timeout(500)
        finally:
            browser.close()
    assert not errors, "\n".join(errors)


def test_finished_run_is_not_polled(base_url, tmp_path):
    """A finished run's view says auto-refresh is paused and fetches nothing more: no tail every 2 s and no
    summary every 10 s (each summary re-reads the run's files on the server)."""
    fakerun = pytest.importorskip("sandbox.tests.fakerun")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    fakerun.make_fake_run(data_dir, "done-run", scores=False)
    errors: list[str] = []
    requests: list[str] = []
    with sync_api.sync_playwright() as p:
        browser = _launch(p)
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            page.on("request", lambda r: requests.append(r.url))
            page.goto(base_url + f"/#/runs/done-run?data_dir={data_dir}")
            page.wait_for_selector(".run-status [data-testid=live-paused]", timeout=15000)
            assert "run finished" in page.inner_text(".run-status [data-testid=live-paused]")
            assert page.locator(".run-status button", has_text="Refresh").count() == 1
            seen = len(requests)
            page.wait_for_timeout(2600)
            later = [u for u in requests[seen:] if "/api/runs/" in u]
            assert not later, later

            # The Refresh button reads the summary once and, the run being finished, stays paused.
            page.click(".run-status button:has-text('Refresh')")
            page.wait_for_timeout(800)
            summaries = [u for u in requests[seen:] if "/api/runs/done-run?" in u]
            assert len(summaries) == 1, summaries
            assert page.locator(".run-status [data-testid=live-paused]").count() == 1
        finally:
            browser.close()
    assert not errors, "\n".join(errors)
