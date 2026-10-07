"""Browser smoke test of the app (sandbox/static/): the real server in a thread, a headless Chromium through
Playwright. Skipped when Playwright or the server cannot be imported, or when no Chromium is installed."""
from __future__ import annotations

import glob
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

            # It validates without errors (warnings are allowed), and is the exact repo study.
            page.wait_for_selector(".status-bar .pill-ok, .status-bar .pill-warn, .status-bar .pill-error", timeout=15000)
            pill = page.inner_text(".status-bar .pill")
            assert "error" not in pill, pill
            assert page.locator(".status-bar .badge-repo").count() == 1
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
