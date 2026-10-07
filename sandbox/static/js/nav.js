// Which view owns a spec path, and jumping to (and flashing) the element that edits it.

export const VIEW_LABELS = {
  study: "Study",
  axes: "Axes",
  personas: "Personas",
  instrument: "Instrument",
  models: "Models",
  run: "Run",
  runs: "Runs",
  jobs: "Jobs",
};

export function viewForPath(path) {
  const head = String(path || "").split(/[.[]/)[0];
  if (head === "factors" || head === "nested" || head === "tables") return "axes";
  if (head === "templates" || head === "derived" || head === "control") return "personas";
  if (head === "instrument") return "instrument";
  if (head === "run") return "models";
  if (head === "randomization") return "run";
  return "study";
}

let pendingFocus = null;

export function goto(view, focusPath) {
  pendingFocus = focusPath || null;
  const target = `#/${view}`;
  if (location.hash === target) {
    window.dispatchEvent(new HashChangeEvent("hashchange"));
  } else {
    location.hash = target;
  }
}

export function gotoIssue(path) {
  goto(viewForPath(path), path || null);
}

export function consumePendingFocus() {
  const p = pendingFocus;
  pendingFocus = null;
  return p;
}

// The element whose data-path is `path` or its longest prefix (at a "." or "[" boundary).
export function findPathElement(root, path) {
  if (!path) return null;
  let best = null;
  let bestLen = -1;
  for (const el of root.querySelectorAll("[data-path]")) {
    const dp = el.dataset.path;
    if (!dp) continue;
    const ok = path === dp || (path.startsWith(dp) && (path[dp.length] === "." || path[dp.length] === "["));
    if (ok && dp.length > bestLen) {
      best = el;
      bestLen = dp.length;
    }
  }
  return best;
}

export function highlightPath(root, path) {
  const el = findPathElement(root, path);
  if (!el) return false;
  const details = el.closest("details");
  if (details) details.open = true;
  el.scrollIntoView({ block: "center", behavior: "smooth" });
  el.classList.remove("flash");
  void el.offsetWidth;
  el.classList.add("flash");
  const focusable = el.matches("input, textarea, select") ? el : el.querySelector("input, textarea, select");
  if (focusable) setTimeout(() => focusable.focus({ preventScroll: true }), 250);
  return true;
}
