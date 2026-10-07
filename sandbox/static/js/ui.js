// DOM helpers. Everything the app shows that came from a file or a model (transcripts, personas, ids) goes
// through textContent via h(); nothing untrusted is ever assigned to innerHTML.

export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  append(el, children);
  if (attrs) applyAttrs(el, attrs);
  return el;
}

export function applyAttrs(el, attrs) {
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") el.className = Array.isArray(v) ? v.filter(Boolean).join(" ") : String(v);
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (k === "text") el.textContent = String(v);
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (["value", "checked", "selected", "disabled", "indeterminate", "readOnly", "open"].includes(k)) el[k] = v;
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, String(v));
  }
  return el;
}

export function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false || c === true) continue;
    el.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export function mount(el, ...children) {
  el.replaceChildren();
  return append(el, children);
}

export function frag(...children) {
  const f = document.createDocumentFragment();
  append(f, children);
  return f;
}

// ---- small widgets -------------------------------------------------------------------------------------

export function button(label, onclick, opts = {}) {
  return h(
    "button",
    {
      type: "button",
      class: ["btn", opts.kind ? `btn-${opts.kind}` : null, opts.small ? "btn-sm" : null, opts.class],
      title: opts.title,
      disabled: opts.disabled,
      onclick,
      "aria-label": opts.ariaLabel,
    },
    label,
  );
}

export function iconButton(symbol, title, onclick, opts = {}) {
  return h(
    "button",
    { type: "button", class: ["icon-btn", opts.class], title, "aria-label": title, disabled: opts.disabled, onclick },
    symbol,
  );
}

export function badge(text, kind = "neutral", title) {
  return h("span", { class: `badge badge-${kind}`, title }, text);
}

const STATUS_KIND = {
  complete: "ok",
  finished: "ok",
  running: "info",
  started: "info",
  failed: "error",
  stopped: "warn",
  error: "error",
  lost: "error",
};

export function statusChip(status) {
  const s = status || "unknown";
  return h("span", { class: `badge badge-${STATUS_KIND[s] || "neutral"} status-chip` }, s);
}

export function progressBar(done, total, opts = {}) {
  const d = Number(done) || 0;
  const t = Number(total) || 0;
  const pct = t > 0 ? Math.max(0, Math.min(100, (d / t) * 100)) : 0;
  const bar = h("div", { class: "progress-fill", style: { width: `${pct}%` } });
  if (opts.kind) bar.classList.add(`progress-${opts.kind}`);
  return h(
    "div",
    { class: "progress", title: opts.title || `${fmtInt(d)} / ${t ? fmtInt(t) : "?"}` },
    h("div", { class: "progress-track" }, bar),
    opts.label === false ? null : h("span", { class: "progress-label" }, `${fmtInt(d)}/${t ? fmtInt(t) : "?"}`),
  );
}

export function details(summary, content, open = false, cls) {
  return h("details", { class: ["details", cls], open }, h("summary", null, summary), content);
}

export function emptyState(text, ...extra) {
  return h("div", { class: "empty" }, h("p", null, text), ...extra);
}

export function kv(pairs, cls) {
  const dl = h("dl", { class: ["kv", cls] });
  for (const [k, v] of pairs) {
    if (v === undefined) continue;
    dl.appendChild(h("dt", null, k));
    dl.appendChild(h("dd", null, v === null ? h("span", { class: "muted" }, "null") : v));
  }
  return dl;
}

export function mono(text, title) {
  return h("code", { class: "mono", title }, text === null || text === undefined ? "—" : String(text));
}

export function shortHash(sha, n = 12) {
  if (!sha) return h("span", { class: "muted" }, "—");
  return h("code", { class: "mono hash", title: String(sha) }, String(sha).slice(0, n));
}

// ---- toasts ----------------------------------------------------------------------------------------------

let toastHost = null;
const liveToasts = new Map();

export function toast(message, opts = {}) {
  if (!toastHost) {
    toastHost = h("div", { class: "toasts", role: "region", "aria-live": "polite", "aria-label": "Notifications" });
    document.body.appendChild(toastHost);
  }
  const kind = opts.kind || "error";
  const key = `${kind}:${message}`;
  if (liveToasts.has(key)) {
    const prev = liveToasts.get(key);
    prev.classList.remove("flash");
    void prev.offsetWidth;
    prev.classList.add("flash");
    return prev;
  }
  const close = () => {
    liveToasts.delete(key);
    el.remove();
  };
  const lines = (opts.detail || []).slice(0, 8);
  const el = h(
    "div",
    { class: `toast toast-${kind}`, role: kind === "error" ? "alert" : "status" },
    h(
      "div",
      { class: "toast-body" },
      h("div", { class: "toast-msg" }, message),
      lines.length ? h("ul", { class: "toast-detail" }, lines.map((l) => h("li", null, l))) : null,
      (opts.detail || []).length > lines.length
        ? h("div", { class: "muted small" }, `… and ${opts.detail.length - lines.length} more`)
        : null,
    ),
    iconButton("×", "Dismiss", close, { class: "toast-close" }),
  );
  liveToasts.set(key, el);
  toastHost.appendChild(el);
  const timeout = opts.timeout ?? (kind === "error" ? 0 : 4000);
  if (timeout > 0) setTimeout(close, timeout);
  return el;
}

export function toastError(err, prefix) {
  if (err && err.name === "AbortError") return;
  const msg = err && err.message ? err.message : String(err);
  const detail = (err && err.issues ? err.issues : []).map((i) =>
    typeof i === "string" ? i : `${i.level || "error"} ${i.path ? i.path + ": " : ""}${i.message || ""}`,
  );
  toast(prefix ? `${prefix}: ${msg}` : msg, { kind: "error", detail });
}

// ---- modal dialogs -----------------------------------------------------------------------------------------

export function modal({ title, body, actions, onOpen, wide }) {
  return new Promise((resolve) => {
    const prevFocus = document.activeElement;
    let done = false;
    const finish = (value) => {
      if (done) return;
      done = true;
      backdrop.remove();
      document.removeEventListener("keydown", onKey, true);
      if (prevFocus && prevFocus.focus) prevFocus.focus();
      resolve(value);
    };
    const onKey = (e) => {
      if (e.key === "Escape") {
        e.preventDefault();
        finish(undefined);
      }
    };
    const btns = (actions || [{ label: "OK", value: true, kind: "primary" }]).map((a) => {
      const b = button(a.label, async () => {
        if (a.validate) {
          const ok = await a.validate();
          if (!ok) return;
        }
        finish(typeof a.value === "function" ? a.value() : a.value);
      }, { kind: a.kind });
      if (a.default) b.dataset.default = "1";
      return b;
    });
    const dialog = h(
      "div",
      { class: ["modal", wide ? "modal-wide" : null], role: "dialog", "aria-modal": "true", "aria-label": title },
      h("div", { class: "modal-head" }, h("h2", null, title), iconButton("×", "Close", () => finish(undefined))),
      h("div", { class: "modal-body" }, body),
      h("div", { class: "modal-actions" }, btns),
    );
    const backdrop = h("div", { class: "modal-backdrop", onmousedown: (e) => e.target === backdrop && finish(undefined) }, dialog);
    document.body.appendChild(backdrop);
    document.addEventListener("keydown", onKey, true);
    dialog.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && e.target.tagName === "INPUT") {
        const def = btns.find((b) => b.dataset.default);
        if (def) {
          e.preventDefault();
          def.click();
        }
      }
    });
    const focusTarget = dialog.querySelector("input, select, textarea") || btns[btns.length - 1];
    if (focusTarget) setTimeout(() => focusTarget.focus(), 0);
    if (onOpen) onOpen(dialog, finish);
  });
}

export function confirmDialog(title, message, okLabel = "OK", kind = "primary") {
  return modal({
    title,
    body: typeof message === "string" ? h("p", null, message) : message,
    actions: [
      { label: "Cancel", value: false },
      { label: okLabel, value: true, kind, default: true },
    ],
  }).then((v) => v === true);
}

export function promptDialog({ title, label, value = "", placeholder, help, validate, okLabel = "OK" }) {
  const input = h("input", { type: "text", class: "input", value, placeholder, spellcheck: "false" });
  const err = h("div", { class: "field-error", role: "alert" });
  const check = () => {
    const msg = validate ? validate(input.value.trim()) : null;
    err.textContent = msg || "";
    return !msg;
  };
  input.addEventListener("input", () => {
    if (err.textContent) check();
  });
  return modal({
    title,
    body: h("label", { class: "field" }, h("span", { class: "field-label" }, label), input, help ? h("span", { class: "field-help" }, help) : null, err),
    actions: [
      { label: "Cancel", value: undefined },
      { label: okLabel, kind: "primary", default: true, validate: check, value: () => input.value.trim() },
    ],
  });
}

// ---- formatting --------------------------------------------------------------------------------------------

export function fmtInt(n) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return "—";
  return Math.round(Number(n)).toLocaleString("en-US");
}

export function fmtNum(n, digits = 2) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return "—";
  return Number(n).toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function fmtSigned(n, digits = 2) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return "—";
  const v = Number(n);
  const s = Math.abs(v).toFixed(digits);
  if (Number(s) === 0) return (0).toFixed(digits);
  return (v > 0 ? "+" : "−") + s;
}

export function fmtTime(ts) {
  if (!ts) return "—";
  const d = typeof ts === "number" ? new Date(ts * (ts < 1e12 ? 1000 : 1)) : new Date(ts);
  if (Number.isNaN(d.getTime())) return String(ts);
  const pad = (x) => String(x).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

export function fmtDuration(seconds) {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "—";
  const hrs = seconds / 3600;
  if (hrs < 1) return `${Math.max(1, Math.round(seconds / 60))} min`;
  if (hrs < 48) return `${hrs.toFixed(1)} h`;
  return `${(hrs / 24).toFixed(1)} days`;
}

export function plural(n, word, pluralWord) {
  return `${fmtInt(n)} ${n === 1 ? word : pluralWord || word + "s"}`;
}

export function deltaClass(d) {
  if (d === null || d === undefined || Number.isNaN(Number(d))) return "delta-none";
  const v = Number(d);
  if (Math.abs(v) < 1e-9) return "delta-zero";
  return v < 0 ? "delta-neg" : "delta-pos";
}

// ---- shell lines ---------------------------------------------------------------------------------------------

export function shellQuote(arg) {
  const s = String(arg);
  if (s === "") return "''";
  if (/^[A-Za-z0-9_\/.,:=@%+-]+$/.test(s)) return s;
  return `'${s.replace(/'/g, `'\\''`)}'`;
}

export function shellLine(argv) {
  return (argv || []).map(shellQuote).join(" ");
}

export async function copyText(text) {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    /* fall through to the textarea trick */
  }
  const ta = h("textarea", { class: "offscreen", readonly: true }, text);
  document.body.appendChild(ta);
  ta.select();
  let ok = false;
  try {
    ok = document.execCommand("copy");
  } catch {
    ok = false;
  }
  ta.remove();
  return ok;
}

export function copyButton(getText, label = "Copy") {
  const b = button(label, async () => {
    const ok = await copyText(typeof getText === "function" ? getText() : getText);
    b.textContent = ok ? "Copied" : "Copy failed";
    setTimeout(() => (b.textContent = label), 1200);
  }, { small: true, class: "copy-btn" });
  return b;
}

export function commandLine(argv, label) {
  const line = Array.isArray(argv) ? shellLine(argv) : String(argv || "");
  return h(
    "div",
    { class: "cmd" },
    label ? h("span", { class: "cmd-label" }, label) : null,
    h("code", { class: "cmd-text" }, line),
    copyButton(line),
  );
}

// ---- timing ---------------------------------------------------------------------------------------------------

export function debounce(fn, ms) {
  let t = null;
  const d = (...args) => {
    clearTimeout(t);
    t = setTimeout(() => {
      t = null;
      fn(...args);
    }, ms);
  };
  d.cancel = () => clearTimeout(t);
  d.flush = (...args) => {
    clearTimeout(t);
    t = null;
    fn(...args);
  };
  return d;
}

// Repeatedly run `fn` every `ms` while `alive()` is true; fn's errors are passed to onError (once per
// distinct message) and do not stop the loop. Returns a stop function.
export function poll(fn, ms, alive, onError) {
  let stopped = false;
  let timer = null;
  let lastErr = null;
  const tick = async () => {
    if (stopped || !alive()) return;
    try {
      const again = await fn();
      lastErr = null;
      if (again === false) return;
    } catch (err) {
      const m = err && err.message;
      if (onError && m !== lastErr) onError(err);
      lastErr = m;
    }
    if (!stopped && alive()) timer = setTimeout(tick, ms);
  };
  timer = setTimeout(tick, 0);
  return () => {
    stopped = true;
    clearTimeout(timer);
  };
}

export function autosize(ta) {
  const fit = () => {
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight + 2, 600)}px`;
  };
  ta.addEventListener("input", fit);
  requestAnimationFrame(fit);
  return ta;
}

export function insertAtCursor(ta, text) {
  if (!ta) return;
  const start = ta.selectionStart ?? ta.value.length;
  const end = ta.selectionEnd ?? ta.value.length;
  ta.setRangeText(text, start, end, "end");
  ta.focus();
  ta.dispatchEvent(new Event("input", { bubbles: true }));
}

export function storageGet(key, fallback) {
  try {
    const v = localStorage.getItem(`sandbox.${key}`);
    return v === null ? fallback : JSON.parse(v);
  } catch {
    return fallback;
  }
}

export function storageSet(key, value) {
  try {
    if (value === undefined) localStorage.removeItem(`sandbox.${key}`);
    else localStorage.setItem(`sandbox.${key}`, JSON.stringify(value));
  } catch {
    /* storage is a convenience only */
  }
}

// Render a long list in pages: the first `pageSize` rows now, a "show more" button for the rest.
export function pagedRows(tbody, items, renderRow, pageSize = 500, colspan = 1) {
  let shown = 0;
  const more = h("tr", { class: "more-row" });
  const fill = () => {
    more.remove();
    const end = Math.min(items.length, shown + pageSize);
    const f = document.createDocumentFragment();
    for (let i = shown; i < end; i++) f.appendChild(renderRow(items[i], i));
    tbody.appendChild(f);
    shown = end;
    if (shown < items.length) {
      mount(
        more,
        h(
          "td",
          { colspan },
          button(`Show ${fmtInt(Math.min(pageSize, items.length - shown))} more (${fmtInt(items.length - shown)} not shown)`, fill, { small: true }),
        ),
      );
      tbody.appendChild(more);
    }
  };
  fill();
  return { get shown() { return shown; } };
}
