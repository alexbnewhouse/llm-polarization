// One dyad of a run: the persona, the transcript as a chat (seeker left, mentor right) with per-message
// cache and timing accounting and judge scores, the mentor's pre/post survey with deltas, and the judge's
// scores across turns. Text from the models goes in through textContent only.

import { api } from "../api.js";
import {
  h, mount, badge, button, statusChip, fmtInt, fmtNum, fmtSigned, fmtTime, details, toastError, deltaClass, debounce, emptyState,
} from "../ui.js";
import { lineChart } from "../charts.js";
import {
  rs, currentDataDir, loadSummary, startLive, levelLabel, runHref, dyadHref, runJobRunning, parseTs, STALL_MS, pausedText, isMockRole, mockBadge,
} from "./runstate.js";

function latestStatus(detail) {
  const rows = (detail.status || []).filter((r) => r.attempt === undefined || r.attempt === detail.attempt);
  const last = rows[rows.length - 1];
  if (!last) return "running";
  return last.status === "started" ? "running" : last.status;
}

function scoreClass(v) {
  if (v === null || v === undefined || !Number.isFinite(Number(v))) return "score-none";
  const x = Number(v);
  return x >= 0.75 ? "score-hi" : x >= 0.5 ? "score-mid" : "score-lo";
}

function scoreChips(rows, multiJudge) {
  return (rows || []).map((s) =>
    h(
      "span",
      {
        class: `chip score-chip ${scoreClass(s.score)}`,
        title: [s.rationale ? `Rationale: ${s.rationale}` : null, s.error ? `Error: ${s.error}` : null, s.judge_sha256 ? `judge ${s.judge_sha256}` : null].filter(Boolean).join("\n") || "no rationale",
      },
      `${s.metric}${multiJudge && s.judge_sha256 ? `·${String(s.judge_sha256).slice(0, 6)}` : ""}: ${s.score === null || s.score === undefined ? (s.error ? "error" : "null") : fmtNum(s.score, 2)}`,
    ),
  );
}

function bubble(t, scores, multiJudge) {
  const seeker = t.agent === "seeker";
  const isErr = t.finish_reason === "error" || !!t.error;
  const tps = t.timings && Number.isFinite(Number(t.timings.predicted_per_second)) ? Number(t.timings.predicted_per_second) : null;
  const meta = [
    t.prompt_n !== undefined && t.prompt_n !== null ? `prompt_n ${fmtInt(t.prompt_n)}` : null,
    t.expected_new !== undefined && t.expected_new !== null ? `expected ${fmtInt(t.expected_new)}` : null,
    t.predicted_n !== undefined && t.predicted_n !== null ? `predicted_n ${fmtInt(t.predicted_n)}` : null,
    t.finish_reason ? `finish ${t.finish_reason}` : null,
    tps !== null ? `${tps.toFixed(1)} tok/s` : null,
    t.id_slot !== undefined && t.id_slot !== null ? `slot ${t.id_slot}` : null,
  ].filter(Boolean);
  return h(
    "div",
    { class: ["msg", seeker ? "msg-seeker" : "msg-mentor", isErr ? "msg-error" : null], dataset: { turn: String(t.turn), agent: String(t.agent) } },
    h(
      "div",
      { class: "msg-head" },
      h("span", { class: "msg-who" }, seeker ? "Seeker" : "Mentor"),
      h("span", { class: "msg-turn muted" }, `turn ${t.turn}`),
      t.cache_warning
        ? badge("⚠ cache", "warn", `The server prefilled ${t.prompt_n ?? "?"} tokens where ${t.expected_new ?? "?"} were expected: the KV cache was not reused for this message.`)
        : null,
      t.truncated ? badge("truncated", "error", "The slot ran out of context") : null,
      t.finish_reason === "length" && !t.truncated ? badge("length", "warn", "Stopped at n_predict") : null,
      isErr ? badge("error", "error") : null,
    ),
    h("div", { class: "msg-text" }, t.text === null || t.text === undefined ? "" : String(t.text)),
    t.error ? h("div", { class: "msg-errtext" }, String(t.error)) : null,
    h("div", { class: "msg-meta" }, meta.join(" · ")),
    scores && scores.length ? h("div", { class: "chips" }, scoreChips(scores, multiJudge)) : null,
  );
}

export function renderDyad(root, ctx, runId, dyadId) {
  const attemptQ = ctx.query.get("attempt");
  const head = h("div", { class: "dyad-head" }, h("h1", { class: "mono" }, dyadId), h("p", { class: "muted" }, "Loading…"));
  const transcript = h("div", { class: "transcript", "aria-label": "Transcript" });
  const side = h("div", { class: "dyad-side" });
  mount(
    root,
    h(
      "nav",
      { class: "crumbs" },
      h("a", { href: "#/runs", class: "link" }, "Runs"),
      h("span", null, " / "),
      h("a", { href: runHref(runId), class: "link mono" }, runId),
      h("span", null, " / "),
      h("span", { class: "mono" }, dyadId),
    ),
    head,
    h("div", { class: "dyad-grid" }, h("div", { class: "dyad-main" }, transcript), side),
  );

  let detail = null;
  let live = null; // the live loop's controller (startLive)
  const liveHost = h("span", { class: "live-note" });
  const seen = new Set();
  let scoresByKey = new Map();
  let lastTurn = 0;

  const indexScores = () => {
    scoresByKey = new Map();
    for (const s of detail.scores || []) {
      if (s.attempt !== undefined && s.attempt !== detail.attempt) continue;
      const k = `${s.turn}|${s.agent}`;
      if (!scoresByKey.has(k)) scoresByKey.set(k, []);
      scoresByKey.get(k).push(s);
    }
  };
  const multiJudge = () => new Set((detail.scores || []).map((s) => s.judge_sha256).filter(Boolean)).size > 1;

  const appendTurns = (rows, liveAppend = false) => {
    const main = document.getElementById("main");
    const nearBottom = main ? main.scrollHeight - main.scrollTop - main.clientHeight < 120 : false;
    let added = 0;
    for (const t of rows) {
      const k = `${t.turn}|${t.agent}`;
      if (seen.has(k)) continue;
      seen.add(k);
      if (Number(t.turn) !== lastTurn) {
        lastTurn = Number(t.turn);
        transcript.appendChild(h("div", { class: "turn-divider" }, h("span", null, `Turn ${t.turn}`)));
      }
      transcript.appendChild(bubble(t, scoresByKey.get(k), multiJudge()));
      added++;
    }
    if (added && liveAppend && nearBottom && main) main.scrollTop = main.scrollHeight;
    const empty = transcript.querySelector(".empty");
    if (empty && seen.size) empty.remove();
    return added;
  };

  const drawHead = () => {
    const d = detail.dyad || {};
    const st = latestStatus(detail);
    const attempts = detail.attempts || [detail.attempt];
    const flag = detail.flag;
    const roles = (rs.summaryRunId === runId && rs.summary && rs.summary.roles) || {};
    const mock = ["seeker", "mentor"].filter((k) => isMockRole(roles[k]));
    mount(
      head,
      h("div", { class: "run-title" }, h("h1", { class: "mono title-id" }, dyadId), statusChip(st), mock.length ? mockBadge(mock) : null, flag ? (flag.flagged ? badge(`flagged at turn ${flag.first_flag_turn ?? "?"}`, "warn", `metric ${flag.metric}, threshold ${flag.threshold}, run length ${flag.run_length}`) : badge("not flagged", "neutral")) : null, liveHost),
      h(
        "div",
        { class: "dyad-meta" },
        h("span", { class: "cond" }, Object.entries(d.condition || {}).map(([k, v]) => h("span", { class: "cond-item" }, h("span", { class: "muted" }, `${k}=`), h("span", { class: "mono" }, levelLabel(v))))),
        h("span", { class: "muted small" }, `mode ${d.persona_mode || "?"} · seed ${d.seed ?? "?"} · ${d.n_turns ?? "?"} turns · started ${fmtTime(d.ts)}`),
        attempts.length > 1
          ? h(
              "span",
              { class: "attempts" },
              h("span", { class: "muted small" }, "attempt"),
              attempts.map((a) =>
                h("a", { class: ["attempt-btn", a === detail.attempt ? "active" : null], href: dyadHref(runId, dyadId, a), title: a === Math.max(...attempts) ? "latest attempt" : "earlier attempt (kept, not analysed unless it is the latest complete one)" }, String(a)),
              ),
            )
          : h("span", { class: "muted small" }, `attempt ${detail.attempt ?? 1}`),
      ),
      (detail.status || []).filter((r) => r.attempt === detail.attempt && r.reason).map((r) => h("div", { class: "error-box" }, h("strong", null, "Failed: "), r.reason)),
    );
  };

  const drawSide = () => {
    const d = detail.dyad || {};
    const pairs = detail.survey_pairs || [];
    const byBattery = {};
    for (const p of pairs) {
      if (p.delta === null || p.delta === undefined || !p.scale) continue;
      const span = Number(p.scale.max) - Number(p.scale.min);
      if (!(span > 0)) continue;
      (byBattery[p.battery] = byBattery[p.battery] || []).push(Number(p.delta) / span);
    }
    const groups = [];
    for (const p of pairs) {
      let g = groups.find((x) => x.battery === p.battery);
      if (!g) groups.push((g = { battery: p.battery, scale: p.scale, items: [] }));
      g.items.push(p);
    }
    const val = (v) => (v === null || v === undefined ? "—" : String(v));
    const survey = pairs.length
      ? h(
          "div",
          null,
          h(
            "div",
            { class: "table-wrap" },
            h(
              "table",
              { class: "table compact survey-table" },
              h("thead", null, h("tr", null, h("th", null, "item"), h("th", { class: "num" }, "pre"), h("th", { class: "num" }, "post"), h("th", { class: "num" }, "Δ"))),
              groups.map((g) => {
                const xs = byBattery[g.battery] || [];
                const m = xs.length ? xs.reduce((a, x) => a + x, 0) / xs.length : null;
                return h(
                  "tbody",
                  null,
                  h(
                    "tr",
                    { class: "group-row" },
                    h("td", null, h("strong", null, g.battery), g.scale ? h("span", { class: "muted small" }, ` ${g.scale.min}–${g.scale.max}`) : null),
                    h("td", { class: "num muted small", colspan: 2, title: "mean of Δ / (max − min) over the battery's items" }, "mean norm. Δ"),
                    h("td", { class: `num delta ${deltaClass(m)}` }, m === null ? "—" : fmtSigned(m, 2)),
                  ),
                  g.items.map((p) =>
                    h(
                      "tr",
                      null,
                      h("td", { class: "mono small item-id", title: p.item_id }, p.item_id),
                      h("td", { class: "num" }, val(p.pre)),
                      h("td", { class: "num" }, val(p.post)),
                      h("td", { class: `num delta ${deltaClass(p.delta)}` }, p.delta === null || p.delta === undefined ? "—" : fmtSigned(p.delta, 0)),
                    ),
                  ),
                );
              }),
            ),
          ),
        )
      : h("p", { class: "muted" }, "No survey pairs yet (the post survey runs after the last turn).");

    const scoreRows = (detail.scores || []).filter((s) => (s.attempt === undefined || s.attempt === detail.attempt) && Number.isFinite(Number(s.score)) && s.score !== null);
    const multi = multiJudge();
    const seriesMap = new Map();
    for (const s of scoreRows) {
      const key = multi ? `${s.metric} · ${String(s.judge_sha256 || "").slice(0, 8)}` : s.metric;
      if (!seriesMap.has(key)) seriesMap.set(key, []);
      seriesMap.get(key).push({ x: Number(s.turn), y: Number(s.score) });
    }
    const series = [...seriesMap.entries()].map(([label, points], i) => ({ key: label, label, colorIndex: i, points }));
    mount(
      side,
      h(
        "section",
        { class: "card" },
        details(
          h("span", null, "Persona", h("span", { class: "muted small" }, ` ${fmtInt((d.persona_text || "").length)} chars`)),
          h("div", null, h("div", { class: "persona-text" }, d.persona_text || ""), h("div", { class: "preview-label" }, "Reminder"), h("div", { class: "persona-text reminder" }, d.persona_reminder || h("span", { class: "muted" }, "(none)"))),
          false,
        ),
      ),
      h("section", { class: "card" }, h("div", { class: "card-head" }, h("h2", null, "Mentor movement"), h("span", { class: "muted small" }, "post − pre")), survey, h("div", { class: "legend-row small" }, h("span", { class: "delta delta-neg" }, "■ negative"), h("span", { class: "delta delta-zero" }, "■ zero"), h("span", { class: "delta delta-pos" }, "■ positive"))),
      h(
        "section",
        { class: "card" },
        h("div", { class: "card-head" }, h("h2", null, "Judge scores"), h("span", { class: "muted small" }, `${scoreRows.length} scored`)),
        series.length ? lineChart({ series, title: "Judge score by turn", xLabel: "turn", yLabel: "score", yDomain: [0, 1], width: 420, height: 240 }) : h("p", { class: "muted" }, "Not scored yet (Actions → Score)."),
      ),
    );
  };

  const drawAll = () => {
    indexScores();
    drawHead();
    drawSide();
    transcript.replaceChildren();
    seen.clear();
    lastTurn = 0;
    const turns = (detail.turns || []).filter((t) => t.attempt === undefined || t.attempt === detail.attempt);
    if (!turns.length) transcript.appendChild(emptyState("No messages yet."));
    appendTurns(turns);
  };

  const load = async (full) => {
    const res = await api.dyad(runId, dyadId, currentDataDir(), attemptQ || undefined);
    if (!ctx.alive()) return;
    const attemptChanged = detail && res.attempt !== detail.attempt;
    detail = res;
    if (full || attemptChanged) drawAll();
    else {
      indexScores();
      drawHead();
      drawSide();
      appendTurns((detail.turns || []).filter((t) => t.attempt === undefined || t.attempt === detail.attempt), true);
    }
    afterLoad();
  };
  const refetch = debounce(() => load(false).catch((err) => toastError(err, `Dyad ${dyadId}`)), 800);
  ctx.cleanup(() => refetch.cancel());

  // The dyad is live while its latest attempt (the one shown) is running and something still writes the run: a
  // run job of this sandbox, or a row newer than STALL_MS (a killed run leaves its dyads at "started").
  const dyadRunning = () => !!detail && latestStatus(detail) === "running" && (!attemptQ || Number(attemptQ) === detail.attempt);
  let lastSeen = 0;
  const dyadLiveness = async () => {
    if (!dyadRunning()) return { live: false, reason: "finished" };
    if (await runJobRunning(runId)) return { live: true, reason: "job" };
    let last = lastSeen;
    for (const r of [...(detail.status || []), ...(detail.turns || [])]) last = Math.max(last, parseTs(r.ts));
    if (!last || Date.now() - last > STALL_MS) return { live: false, reason: "stalled", since: last || null };
    return { live: true, reason: "running" };
  };
  const drawLive = () => {
    if (!live || !rs.auto) return mount(liveHost);
    if (!live.paused) return mount(liveHost, live.running ? badge("live", "info", "Auto-refresh: new messages appear as they are written") : null);
    mount(
      liveHost,
      h("span", { class: "live-paused muted small", "data-testid": "live-paused" }, pausedText(live, "dyad")),
      button("Refresh", async () => {
        try {
          await load(false);
          if (live) await live.refresh();
        } catch (err) {
          toastError(err, `Dyad ${dyadId}`);
        }
        drawLive();
      }, { small: true, title: "Read the dyad again; auto-refresh resumes if it is still running" }),
    );
  };
  // After every re-read: a dyad that finished stops the loop.
  const afterLoad = () => {
    if (live && !live.paused && !dyadRunning()) live.pause("finished");
    drawLive();
  };

  (async () => {
    try {
      await load(true);
    } catch (err) {
      if (!ctx.alive()) return;
      toastError(err, `Could not open dyad ${dyadId}`);
      mount(head, h("h1", { class: "mono" }, dyadId), h("div", { class: "error-box" }, err.message, err.status === 404 ? ` (data dir: ${currentDataDir()})` : ""), h("p", null, h("a", { class: "link", href: runHref(runId) }, `← back to ${runId}`)));
      return;
    }
    // Live: tail the run's files when the server gives offsets, else re-read this dyad every 3 s; either way
    // only while the dyad is live (a finished dyad is never polled; Refresh reads it again).
    try {
      if (rs.summaryRunId !== runId || !rs.summary) await loadSummary(runId);
    } catch {
      /* the summary only provides offsets and the models; without it we fall back to polling the dyad */
    }
    if (!ctx.alive()) return;
    drawHead();
    const tailing = !!(rs.summaryRunId === runId && rs.summary && rs.summary.offsets);
    live = startLive(ctx, runId, {
      summary: false, // this view needs no run summary; liveness is the dyad's own
      liveness: dyadLiveness,
      onState: drawLive,
      isOn: () => rs.auto && dyadRunning(),
      interval: tailing ? 2000 : 3000,
      onTick: tailing ? null : () => load(false),
      onRows: (rows) => {
        const mine = (list) => (list || []).filter((r) => r.dyad_id === dyadId && (r.attempt === undefined || r.attempt === detail.attempt));
        const turns = mine(rows.turns);
        for (const r of [...turns, ...mine(rows.status)]) lastSeen = Math.max(lastSeen, parseTs(r.ts));
        if (turns.length) appendTurns(turns, true);
        const newAttempt = (rows.status || []).some((r) => r.dyad_id === dyadId && r.attempt > detail.attempt);
        if (newAttempt && !attemptQ) {
          load(true).catch((err) => toastError(err, `Dyad ${dyadId}`));
          return;
        }
        if (mine(rows.status).length || mine(rows.surveys).length || mine(rows.scores).length) refetch();
      },
    });
    ctx.cleanup(() => live && live.stop());
    drawLive();
  })();
}
