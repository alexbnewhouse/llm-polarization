// Models view: seeker, mentor and judge endpoints, sampling, and the run's operational settings; a button that
// starts the demo mock backend and points the study at it (synthetic text, never for data).

import { api } from "../api.js";
import { h, mount, button, badge, toast, toastError, mono } from "../ui.js";
import { state, touch, refreshMock, rememberMock, mockRoles, defaultRun, mockDataDir } from "../state.js";
import { field } from "../forms.js";
import { viewHeader, card, issuePanel, mockDataDirWarning } from "./common.js";

const ROLE_HELP = {
  seeker: "Plays the persona. One model across every arm; a different family from the mentor.",
  mentor: "The zero-shot advisor whose survey movement is the outcome. Must be a different server from the seeker.",
  judge: "Only used by score. Never the seeker or the mentor, never the mentor's family.",
};

function roleCard(role) {
  const p = ["run", role];
  return h(
    "section",
    { class: "card role-card", dataset: { path: `run.${role}` } },
    h("div", { class: "card-head" }, h("h2", null, role[0].toUpperCase() + role.slice(1))),
    h("p", { class: "card-help" }, ROLE_HELP[role]),
    field("url", [...p, "url"], { mono: true, placeholder: "http://127.0.0.1:8201", help: "A llama-server started with --jinja" }),
    field("gguf_path", [...p, "gguf_path"], { mono: true, nullable: true, placeholder: "(read from the server's /props)", optional: true, help: "Hashed into the manifest; blank = the path the server reports" }),
    role === "mentor"
      ? h("p", { class: "note" }, "Nothing is ever sent to the mentor beyond its chat template: no mentor system prompt, by design (any system text on the mentor is treatment).")
      : null,
  );
}

function mockCard(ctx) {
  const host = h("div");
  const draw = () => {
    const m = state.mock;
    const using = mockRoles();
    mount(
      host,
      h(
        "div",
        { class: "mock-status" },
        m && m.running ? badge("mock running", "warn") : badge("mock stopped", "neutral"),
        using.length ? h("span", { class: "warn-text" }, `This study points at the mock for: ${using.join(", ")}`) : h("span", { class: "muted small" }, "This study does not point at the mock."),
      ),
      m && m.running
        ? h(
            "table",
            { class: "table compact" },
            h("thead", null, h("tr", null, h("th", null, "role"), h("th", null, "url"), h("th", null, "gguf_path"))),
            h(
              "tbody",
              null,
              ["seeker", "mentor", "judge"].map((r) => h("tr", null, h("td", null, r), h("td", null, mono(m[r] && m[r].url)), h("td", null, mono(m[r] && m[r].gguf_path)))),
            ),
          )
        : null,
      h("p", { class: "muted small" }, `Start also sets data_dir to ${mockDataDir()}, so demo runs stay out of the repo's data/.`),
      h(
        "div",
        { class: "card-foot" },
        button("Start demo mock servers", async () => {
          try {
            const info = await api.mockStart({});
            state.mock = info;
            rememberMock(info);
            const run = state.spec.run || (state.spec.run = defaultRun());
            for (const r of ["seeker", "mentor", "judge"]) {
              if (!info || !info[r]) continue;
              run[r] = run[r] || {};
              run[r].url = info[r].url;
              run[r].gguf_path = info[r].gguf_path ?? null;
            }
            // Demo runs never go into data/ (its manifest.json files are git-trackable): the sandbox's own dir.
            const dd = mockDataDir();
            run.data_dir = dd;
            touch();
            toast(`Mock servers started; seeker, mentor and judge now point at them, and data_dir is now ${dd} so demo runs stay out of data/. Synthetic text only.`, { kind: "warn", timeout: 8000 });
            ctx.rerender();
          } catch (err) {
            toastError(err, "Could not start the mock servers");
          }
        }, { kind: "primary", title: "POST /api/mock/start: three fake llama-servers with weightless GGUFs" }),
        m && m.running
          ? button("Stop mock servers", async () => {
              try {
                state.mock = await api.mockStop();
              } catch (err) {
                toastError(err, "Could not stop the mock servers");
              }
              await refreshMock();
              ctx.rerender();
            })
          : null,
        button("Refresh", async () => {
          await refreshMock();
          draw();
        }, { small: true }),
      ),
    );
  };
  ctx.on("mock", draw);
  ctx.on("changed", draw);
  draw();
  return card(
    "Demo mock backend",
    { help: "Fake llama-servers that speak the same API (props, apply-template, tokenize, completion with a per-slot cache) so the whole pipeline runs without a GPU. Replies are deterministic filler." },
    h("div", { class: "banner banner-mock inline" }, h("strong", null, "MOCK — synthetic text, never for data.")),
    host,
  );
}

export function render(root, ctx) {
  const spec = state.spec;
  if (!spec.run || typeof spec.run !== "object") {
    mount(root, viewHeader("Models"), card("No run block", button("Create from defaults", () => {
      spec.run = defaultRun();
      touch();
      ctx.rerender();
    }, { kind: "primary" })));
    return;
  }
  const ggufWarn =
    state.meta && state.meta.gguf_importable === false && !spec.run.gguf_py_path
      ? h("p", { class: "warn-text" }, "gguf-py is not importable by the server's Python: set gguf_py_path (llama.cpp's gguf-py directory) so the harness can read chat templates out of the GGUFs.")
      : null;
  mount(
    root,
    viewHeader("Models", "The harness config for this study, minus batteries and grid (export fills those in)."),
    issuePanel(ctx, ["run"]),
    mockDataDirWarning(ctx),
    h("div", { class: "role-grid" }, ["seeker", "mentor", "judge"].map(roleCard)),
    card(
      "Generation (dialogue turns)",
      { path: "run.generation", help: "Surveys and the judge use their own fixed settings (temperature 0; n_predict 32 and 160), written onto their rows." },
      h(
        "div",
        { class: "form-row" },
        field("temperature", "run.generation.temperature", { type: "float", step: "0.05" }),
        field("top_p", "run.generation.top_p", { type: "float", step: "0.01" }),
        field("n_predict", "run.generation.n_predict", { type: "int", help: "Max tokens per message" }),
        field("timeout", "run.generation.timeout", { type: "int", help: "Seconds per request" }),
        field("enable_thinking", "run.generation.enable_thinking", { type: "checkbox", help: "Passed to templates that support it" }),
      ),
    ),
    card(
      "Run",
      { path: "run" },
      h(
        "div",
        { class: "form-row" },
        field("run_seed", "run.run_seed", { type: "int", help: "Every generation seed derives from this and the dyad seed. Record it." }),
        field("now", "run.now", { mono: true, placeholder: "YYYY-MM-DD", help: "Date fed to templates that print one. Freeze it for the study." }),
        field("data_dir", "run.data_dir", { mono: true, help: "Where data/<run_id>/ is written" }),
      ),
      h(
        "div",
        { class: "form-row" },
        field("concurrency", "run.concurrency", { type: "int", nullable: true, placeholder: "(slots of the smaller server)", help: "Dialogues at once; blank = null" }),
        field("cache_reuse_limit", "run.cache_reuse_limit", { type: "int", nullable: true, placeholder: "(disabled)", help: "Tokens; must exceed 2 × n_predict + reminder. Blank = null disables." }),
        field("gguf_py_path", "run.gguf_py_path", { mono: true, nullable: true, placeholder: "(gguf importable)", help: "llama.cpp's gguf-py directory; blank = null" }),
      ),
      ggufWarn,
    ),
    mockCard(ctx),
  );
}
