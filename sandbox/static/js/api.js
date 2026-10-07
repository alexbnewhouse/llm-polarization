// Thin fetch wrappers over the sandbox JSON API (spec section 5). Every non-GET request carries
// X-Sandbox: 1 and a JSON content type; the server refuses anything else. Errors become ApiError with
// the server's {"error", "issues"} so callers can show both.

export class ApiError extends Error {
  constructor(status, message, issues, body) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.issues = Array.isArray(issues) ? issues : [];
    this.body = body;
  }
}

function qs(query) {
  if (!query) return "";
  const parts = [];
  for (const [k, v] of Object.entries(query)) {
    if (v === undefined || v === null || v === "") continue;
    parts.push(`${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`);
  }
  return parts.length ? `?${parts.join("&")}` : "";
}

const seg = (s) => encodeURIComponent(String(s));

export async function request(method, path, { query, body, signal } = {}) {
  const init = { method, headers: { Accept: "application/json" }, signal };
  if (method !== "GET") {
    init.headers["Content-Type"] = "application/json";
    init.headers["X-Sandbox"] = "1";
    init.body = JSON.stringify(body === undefined ? {} : body);
  }
  let res;
  try {
    res = await fetch(path + qs(query), init);
  } catch (err) {
    if (err && err.name === "AbortError") throw err;
    throw new ApiError(0, `Cannot reach the sandbox server (${method} ${path}): ${err && err.message ? err.message : err}`);
  }
  const text = await res.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = null;
    }
  }
  if (!res.ok) {
    const msg = (data && typeof data.error === "string" && data.error) || `${method} ${path} failed: HTTP ${res.status}`;
    throw new ApiError(res.status, msg, data && data.issues, data);
  }
  if (data === null && text) throw new ApiError(res.status, `${method} ${path}: the server did not return JSON`);
  return data;
}

const get = (path, query, opts) => request("GET", path, { query, ...(opts || {}) });
const post = (path, body, opts) => request("POST", path, { body, ...(opts || {}) });
const put = (path, body, opts) => request("PUT", path, { body, ...(opts || {}) });

export const api = {
  meta: () => get("/api/meta"),

  studies: () => get("/api/studies"),
  study: (name) => get(`/api/studies/${seg(name)}`),
  saveStudy: (name, spec) => put(`/api/studies/${seg(name)}`, { spec }),

  validate: (spec, opts) => post("/api/study/validate", { spec }, opts),
  cells: (spec, opts) => post("/api/study/cells", { spec }, opts),
  render: (spec, kind, condition, variant, opts) =>
    post("/api/study/render", { spec, kind, condition, variant: variant ?? null }, opts),
  manifest: (spec, limit = 20, opts) => post("/api/study/manifest", { spec, limit }, opts),
  exportStudy: (spec) => post("/api/study/export", { spec }),
  repoFiles: (spec) => post("/api/study/repo-files", { spec }),

  runs: (dataDir) => get("/api/runs", { data_dir: dataDir }),
  run: (runId, dataDir) => get(`/api/runs/${seg(runId)}`, { data_dir: dataDir }),
  dyad: (runId, dyadId, dataDir, attempt) =>
    get(`/api/runs/${seg(runId)}/dyads/${seg(dyadId)}`, { data_dir: dataDir, attempt }),
  tail: (runId, dataDir, offsets) => get(`/api/runs/${seg(runId)}/tail`, { data_dir: dataDir, ...(offsets || {}) }),
  analysis: (runId, dataDir, { factor, metric, judge } = {}) =>
    get(`/api/runs/${seg(runId)}/analysis`, { data_dir: dataDir, factor, metric, judge }),
  runConfig: (runId, dataDir, judge) => {
    const body = { data_dir: dataDir };
    if (judge) body.judge = judge;
    return post(`/api/runs/${seg(runId)}/config`, body);
  },

  jobs: () => get("/api/jobs"),
  startJob: (fields) => post("/api/jobs", fields),
  jobLog: (jobId, offset = 0) => get(`/api/jobs/${seg(jobId)}/log`, { offset }),
  stopJob: (jobId) => post(`/api/jobs/${seg(jobId)}/stop`, {}),

  mock: () => get("/api/mock"),
  mockStart: (opts) => post("/api/mock/start", opts || {}),
  mockStop: () => post("/api/mock/stop", {}),
};
