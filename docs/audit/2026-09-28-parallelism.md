# Parallelism and concurrency review: llm-polarization

Repo: `/home/user/llm-polarization`, branch `claude/wizardly-goodall-2xnphr`, HEAD `d6a07ec`. Line numbers
are for HEAD. Two commits landed while this review was running (`f63a6a2` docs/comments, `d6a07ec` gap audit).
The harness changes in them are docstrings only, so the behaviour reviewed here is unchanged.
`docs/audit/2026-09-28-gap-audit.md` already covers some of this ground (F7, F8, F15, F17). Where a finding
overlaps it, this report says so and adds the measured evidence.

llama.cpp behaviour is cited from the source at tag **b10488** (the study's build, RUN_APPROACH.md:4),
fetched to `scratchpad/b10488/`.

The experiments ran against a fake llama-server, `scratchpad/fake_llama.py`. It models per-slot prefix
caching and the same deferral as llama.cpp: a request pinned to a busy slot waits for that slot.
`--wrap` mirrors b10488's id_slot wrap-around. Latency is proportional to prefill and decode token counts.
`scratchpad/hrun.py` runs the real harness CLI (a copy in `scratchpad/pkg`) with the GGUF read and the git
calls stubbed. Baseline: `python3 -m pytest harness/tests -q` gives 204 passed, 2 skipped.

## Answers in brief

1. **Model.** The harness uses threads, not asyncio: `ThreadPoolExecutor(max_workers=concurrency)`
   (run.py:465). Concurrency is enforced by the pool size plus a slot free-list (run.py:439-462).
   - Each output file has one `JsonlWriter` per process with its own `threading.Lock`. Each row is a single
     `write()` of a complete line followed by `flush()` (log.py:39-45), so rows cannot interleave within a
     process. There is no fsync.
   - Within a dyad, `status` is written after that dyad's turns and surveys, from the same thread.
2. **Slots.** `concurrency: null` is read from `/props total_slots` (run.py:422-423), falling back to 1 when
   the server does not report it. A configured value is not checked against it (see M1).
   - Every completion request carries `id_slot`, and a dyad keeps its slot for life. So llama.cpp's
     LCP/LRU slot selection is never used, and two dyads with the same persona cannot be moved onto each
     other's slot by the server.
   - Collisions come from somewhere else: a second client of the same server (H1), or `concurrency` above
     the server's slot count (M1).
3. **Ctrl-C.** The spec is enforced: in-flight dyads finish, queued ones never start, and `run` exits 130
   (E5). The operational problems are covered in M3.
4. **Retries.** Within one process, retries are race-free (one `(spec, attempt)` per dyad). Across
   processes there is no lock of any kind (H1).
5. **Throughput.** The harness side is not the bottleneck: no global lock is held across HTTP, and
   `/tokenize` does not queue behind decode. The problems are:
   - The 8-way cap is shared across both servers, so each server averages about 4 active streams (M2).
   - `score` is fully serial (H2).
6. **Benchmarks.** `parallel_scaling.csv` agrees with RUN_APPROACH.md and README.md on every row. The
   benchmark measures a different load shape from the one the harness produces (M2, L5).
7. **Diagnostics.**
   - Recorded: `prompt_n`, `tokens_cached`, `tokens_evaluated`, `truncated`, full `timings` and the
     requested `id_slot` on turn rows.
   - Not recorded: the server's actual slot, the request start time, in-flight load, and the build on
     resume (M4, L2).

---

## Findings, by severity

### H1. Any second client of a server corrupts the run: there is no run lock and no slot ownership

Every `run`, `check`, `survey` and `score` process pins slots starting at 0 (run.py:303, 344, 439, 507,
531). Nothing stops two of them from sharing a server or a `run_id`. `write_manifest` is not atomic across
processes (log.py:102-110; acknowledged in REPRODUCIBILITY.md:346).

**E3: two `run` processes on the same run_id at the same time** (12 dyads, 6 turns):
```
$ (hrun.py run ... --run-id r1 &) ; (hrun.py run ... --run-id r1 &) ; wait
A: run r1: 12 of 12 dyads to run, concurrency 8 / done: 12 complete, 0 failed / exit 0
B: run r1: 12 of 12 dyads to run, concurrency 8 / done: 11 complete, 1 failed / exit 2
status rows: 48  (dyad,attempt,status) written twice: 23
turn rows: 288 (expected 144)  duplicate turn keys: 144  cache_warning rows: 10
   d04 1 d04 turn 6 mentor: KV cache reuse lost: prefilled 1661 tokens on slot 3, expected about 302
seeker: collisions 61 ; mentor: collisions 509          (requests arriving at a busy pinned slot)
resume_index d04 -> {'attempt': 1, 'status': 'failed'}  next_attempt -> 2
```
- Every turn key is duplicated, and the two copies cannot be told apart.
- The `failed` row, written after the other process's `complete`, wins the `>=` tie in `resume_index`
  (log.py:120). A completed attempt 1 is therefore re-queued as attempt 2.
- `Scorer` would score both copies, because the `done` set is built once, before the loop (scorer.py:244, 258).

**E4: `harness check` run while a wave is in progress.** The probe writes about 530 tokens into slot 0 on
both servers (run.py:139-144):
```
check: ok cache_reuse second call prefilled 9 tokens ... on slot 0
run:   done: 7 complete, 1 failed
       d07 turn 4 mentor: KV cache reuse lost: prefilled 1057 tokens on slot 0, expected about 302 (limit 1000)
```
- `survey` re-administration (mentor slot 0) and a second arm's `run` against the shared fixed-seeker
  server (every arm uses the same seeker model: RUN_APPROACH.md:93-95) behave the same way.
- Wherever `cache_reuse_limit` does not trigger, the only symptom is silent cache thrash.
- Overlap: gap audit F9 asks for a study-level lock. It does not describe these slot collisions.

**Fix:**
- Take an exclusive `fcntl.flock` on `data/<run_id>/.lock` for the life of `run`, `survey` and `score`.
- In `check` and at `run` start, read `GET /slots` (enabled by default at b10488, common.h:653). Refuse or
  skip the probe if any slot is processing or holds tokens that this process did not put there.
- Optionally add a `slot_base` config so that two processes can share a server on disjoint slots.

### H2. `score` is strictly serial on slot 0, and its target order defeats the prefix cache

- `cmd_score` builds one judge on slot 0 (run.py:531), and `Scorer.score_run` loops with one blocking
  request at a time (scorer.py:258-298).
- `select_targets` returns targets in `turns.jsonl` order. That order interleaves the 8 concurrent dyads
  (thread timing), and within a row it cycles through the metrics, whose prompts differ from the very
  first line (`TASK:`, scorer.py:68). Consecutive judge prompts therefore share almost no prefix.

**E7** (16 dyads x 8 turns, pilot scope, fake judge with 8 slots):
```
current:                    requests 386  max_active 1  slots used [0]  prefill_tokens 86606  wall 17.9 s
grouped (dyad,metric,turn): requests 386  max_active 1                  prefill_tokens 57966  wall 16.3 s
turns.jsonl order: ['d02','d01','d00','d06','d03','d04','d05','d07','d02','d01', ...]
```
- In this toy the saving is 33%. With real prompts it is much larger. `line_to_line` includes every earlier
  own line (scorer.py:277), about 5k tokens on average and up to about 10k at turn 40. When targets are
  grouped by (dyad, metric) on a pinned slot, each call prefills only the lines added since the previous
  scored turn.

**Wall-clock impact** (assumptions, to be measured in the pilot): `main` scope is 20 calls per dyad, about
77k prefill tokens and 1k generated tokens per dyad, uncached.

| Judge class | Rough speed | Serial time per dyad | Days per arm (2,970 dyads) |
|---|---|---|---|
| MoE, qwen3.6-class | ~1000 tok/s prefill, ~55 tok/s decode | ~95 s | ~3.3 |
| Dense 27B (the `config.example.json` judge on :8099) | ~275 tok/s prefill, ~10 tok/s decode | ~380 s | ~13 |

- For comparison, RUN_APPROACH.md:99-106 gives 4-12 dialogue-days per arm at 40 turns. Judge time appears
  in no budget.
- Grouping plus 8 pinned workers should cut judge time by roughly 3-6x.
- Overlap: gap audit F7 (serial and unbudgeted). The ordering and prefix-cache point is new here.

**Fix:**
- Sort targets by `(dyad_id, attempt, metric, turn)`.
- Run groups on a `ThreadPoolExecutor`, one pinned slot per worker, using the same free-list pattern as
  `cmd_run`.
- Add a `done.add(key)` after each write.
- Budget judge time in RUN_APPROACH.md.

### M1. `concurrency` above the server's slot count is accepted, and b10488 silently wraps the slot id

`concurrency = cfg["concurrency"] or server_slots` (run.py:423) is never compared with `total_slots`. At
b10488, `get_slot_by_id` does `id_slot = id_slot % slots.size()` ("allow id_slot to be out of bounds (wrap
around)", `b10488/server-context.cpp:1460-1462`), so two dyads end up sharing one slot.

**E2b** (`concurrency: 10` against 8 slots, fake with `--wrap`):
```
run r1: 10 of 10 dyads to run, concurrency 10
done: 6 complete, 4 failed          exit 2
d01 turn 4 mentor: KV cache reuse lost: prefilled 1057 tokens on slot 8, expected about 302
d00 ... on slot 9 ; d09 ... on slot 0 ; d08 ... on slot 1
mentor: collisions 48
```
- The rows record `id_slot` 8 and 9, which do not exist on the server.
- `free.pop()` is last-in-first-out, so the first two jobs get slots 9 and 8. A resume at the same
  `concurrency` repeats the failure.
- The `total_slots or 1` fallback (run.py:422) silently drops a server without `total_slots` to
  concurrency 1.

**Fix:**
- Refuse `concurrency > min(total_slots)`, and refuse a missing `total_slots` when `concurrency` is null.
- Log `raw["id_slot"]` (the slot the server actually used, `b10488/server-task.cpp:1058`) next to the
  requested one.

### M2. The operating point assumes 8 active streams per server; the harness gives each server about 4

A dyad is only ever in one server at a time: seeker, then mentor, then seeker again. So
`concurrency = min(slots) = 8` means at most 8 generations across **both** servers.

**E1** (16 dyads x 8 turns, 8 slots per server, seeker and mentor equally fast):
```
run r1: 16 of 16 dyads to run, concurrency 8   done: 16 complete   wall 12.5 s
seeker: mean_active 3.42  max_active 8
mentor: mean_active 4.25  max_active 8       (mentor also carries 26 survey requests per dyad)
```
The load balances itself: a slower mentor holds more of the 8. But RUN_APPROACH.md:99-106 prices each arm
at the np=8, all-8-active aggregate. The same CSV at np=4 gives:

| Arm | np=8 | np=4 | Change |
|---|---|---|---|
| qwen3.6 | 116.4 | 107.2 | -8% |
| gpt-oss | 83.4 | 82.5 | -1% |
| Olmo | 41.5 | 41.4 | 0% |
| glm HIP | 29.1 | 25.6 | -12% |

Other ways `bench_parallel.py` differs from the harness's request pattern:
- **Warm prefill.** The benchmark prefills 5-8 tokens (bench_parallel.py:133-135; RUN_APPROACH.md:26).
  A real turn prefills the partner's message, and in reinforced mode also the seeker's own previous reply
  plus the reminder, because the reminder sits before them in the cached sequence (transcript.py:62-63).
  That is about 300-700 tokens, or about 3-6% of turn time.
- **Turn length.** The benchmark generates 200 tokens per turn (`--gen 200`); the harness caps turns at
  `n_predict` 300 (config.example.json:8). If turns run near the cap, generation costs up to 1.5x.
  Overlap: gap audit F15.
- **Depth.** The benchmark decodes every stream at 32k depth, in lockstep. Real dyads are staggered, with a
  mean depth of about 10k. The CSV at 8k vs 32k (qwen np=8 f16: 134.0 vs 101.3) suggests this makes the
  benchmark 15-25% conservative.
- **Two servers.** The benchmark runs one server. Two servers sharing one GPU were never measured
  (overlap: gap audit F15).
- **Surveys.** The mentor's 26 grammar-constrained survey requests per dyad are not in the benchmark.

**Net:** for the three remaining arms the dialogue-days error is probably within ±15%. The `n_predict` cap
and judge time (H2) are larger.

**Fix:** benchmark the harness itself. Run a 16-dyad pilot manifest through `run` against the real
servers, and compute aggregate tok/s per server from `turns.jsonl` (`predicted_n`, `timings`, `ts`).
Recompute the budget from the np≈4 numbers and the measured mean `predicted_n`.

### M3. Ctrl-C works as written but is hard to operate, and the worker handler is dead code

**E5** (16 dyads, SIGINT at t=3 s):
```
1 SIGINT:  exit=130  seconds from first SIGINT to exit=6.6
           done: 8 complete, 0 failed, 8 not run / stopped by Ctrl-C; re-run with --run-id r1 to resume
2 SIGINTs: exit=-2 (SIGINT)  seconds to exit=6.6   traceback in ex.shutdown -> t.join(); no summary line
3 SIGINTs: same; the in-flight dyads still ran to 'complete'
```
- Python delivers SIGINT only to the main thread, so `except KeyboardInterrupt` inside `job`
  (run.py:455-459) never runs in production. `test_ctrl_c_stops_submitting_new_dyads_and_exits_130` raises
  inside the worker, so it exercises that dead path (overlap: gap audit F17).
- A second Ctrl-C cannot abandon the in-flight dyads. After the traceback, the interpreter's atexit hook
  joins the non-daemon pool threads, and the process keeps running with no message.
- Real in-flight dyads last about 25 min (qwen: 80 messages x ~19 s) to about 1 h (slower arms). An
  operator will reach for `kill -9`.
- A killed dyad is left at `started` and becomes attempt 2 on resume. REPRODUCIBILITY.md:234-238 asks for
  `attempt > 1` to be reported as a failure-selection statistic, and operator kills would inflate it.

**Fix:**
- On the first SIGINT, print "waiting for N in-flight dyads; Ctrl-C again to abandon them".
- On the second, write `status: interrupted` for each in-flight (dyad, attempt) and `os._exit(130)`.
- Have analysis treat `interrupted` separately from `failed`.
- Test the real path by sending `signal.SIGINT` to the main thread (`signal.raise_signal`) with
  concurrency > 1.

### M4. A resume accepts a different llama.cpp build or slot configuration without recording it

`_verify_identity` compares only `model_sha256` and `template_sha256` (run.py:370). `manifest.json` is
never rewritten, and rows carry no build.

**E8** (resume after a server restart with a different `build_info`):
```
run r1: 4 of 10 dyads to run, concurrency 8   done: 4 complete   exit 0
manifest build_info: fake-b0 fake-b0
turn row keys mentioning build: []
attempt-2 completes: ['d00', 'd01', 'd08', 'd09']
```
- REPRODUCIBILITY.md:30 (standard row 8) claims the build is recorded. It is recorded only as it was at the
  first start.
- The watchdog wrappers restart a crashed server in a loop (serve-bulk.sh:23-29). A restart wipes every
  slot cache mid-run. The in-run guard fails those dyads loudly, which is correct, but nothing records the
  restart.
- Overlap: gap audit F8.

**Fix:** on every `run` start, write `data/<run_id>/session-<ts>.json` with both servers' `/props`
(`build_info`, `total_slots`, `n_ctx`) and the concurrency. Refuse a changed `build_info`, or stamp a
`session` id on every row.

### L1. No fsync, and a torn tail line blocks resume

- `JsonlWriter.write` flushes but does not fsync (log.py:39-45).
- `read_jsonl` parses every line strictly (log.py:54).
- Ctrl-C and SIGKILL cannot tear a line, because each row goes out in a single `write()`. Power loss,
  ENOSPC or a host crash can.

**E6** (40 bytes truncated from the end of `status.jsonl`):
```
error: Unterminated string starting at: line 1 column 60 (char 59)      exit 1
```
- The error names no file or line, and the next append would be glued onto the torn row.
- Across files, without fsync, a `complete` row in `status.jsonl` does not guarantee that the dyad's
  `turns.jsonl` rows are on disk.

**Fix:**
- In `read_jsonl`, detect an unterminated final line, report it with its path, and offer to truncate it.
- `os.fsync` `turns` and `surveys` once, before writing `complete`: one call per dyad.

### L2. Some per-request diagnostics for batching effects are missing

Recorded on each turn row: requested `id_slot`, `prompt_n`, `tokens_cached`, `tokens_evaluated`,
`truncated`, and full `timings` (dialogue.py:115-140).

Missing:
- The server-side slot (`raw["id_slot"]`), which differs from the requested one under wrap-around (M1).
- A request start time. `ts` has 1-second resolution and marks completion only (log.py `now_iso`).
- The number of in-flight requests to that server, so batch co-membership cannot be reconstructed.
- Per-row build (M4).
- Survey rows keep only `prompt_n` (survey.py:105-107); score rows keep no server accounting at all.

REPRODUCIBILITY.md:219-222 only declares batching nondeterminism, so this is not a broken promise. It is,
however, the data that would be needed to test it.

Two doc statements are inaccurate:
- REPRODUCIBILITY.md:219 says "Eight dialogues share one llama-server". In fact eight dialogues share two
  servers, and each server's batch is the subset currently on that side, about 4 (E1), plus survey
  requests.
- REPRODUCIBILITY.md:246-250 says "Nothing is half-written". That holds for Ctrl-C but not for a crash
  (L1).

**Fix:** add `t_send`/`t_recv` (epoch in ms), `inflight_at_send` (a per-server counter under the existing
lock), `server_id_slot`, and `tokens_cached` on survey rows.

### L3. The host prompt cache is on for the non-Olmo arms, and it cannot help pinned slots

At b10488, the defaults are `cache_ram_mib = 8192` and `cache_idle_slots = true` (common.h:612-615). Two
paths save into it:
- On each new task, every idle slot is saved to host RAM (server-context.cpp:2353-2369).
- A pinned slot whose new prompt keeps less than half of its old content (`f_keep < 0.5`, which happens at
  every new dyad) triggers `prompt_save`/`prompt_load` (server-context.cpp:1486-1600).

Pinned dialogues never read these entries back. The costs are:
- up to 8 GiB of unified memory per server, and 2 servers per arm, on a box that has already OOM-killed
  resident tiers (RUN_APPROACH.md:149-153);
- a state copy on the server's main loop for roughly every turn;
- exactly the code path that crashes on sliding-window models (the Olmo trace).

The operating-point command (RUN_APPROACH.md:15-18) adds `--cache-ram 0` only for Olmo.

The stall cost is estimated from the source, not measured: roughly one full-slot state copy per turn per
server, likely under 1% of time.

**Fix:** use `--cache-ram 0` for every arm, and note it as part of the operating point.

### L4. The `check` cache probe does not represent the in-run pattern

`_cache_reuse_probe` (run.py:132-149) sends a strict extension on slot 0 of an idle server. The reinforced
seeker is different: each turn diverges about 2 messages plus the reminder before the end of the cached
sequence (transcript.py:62-63). On sliding-window or recurrent seekers, reuse then depends on context
checkpoints (`n_ctx_checkpoints = 32`, b10488 common.h:613; restore logic at server-context.cpp:3197-3290).
If those fail, every dyad dies at turn 2 with `CacheReuseLost`: loud, but at pilot time rather than at
`check`.

Under load the probe is representative, because pinned ids bypass LCP selection. Slots 1..N-1 are never
probed.

**Fix:** add a second probe that appends and then removes a trailing system message around a
600-token assistant/user pair, and assert that `prompt_n` is at most that pair plus the margin.

### L5. The benchmark's "per-stream tok/s" is the server-side decode rate

`warm_per_stream_tps` is the mean of `predicted_per_second` (bench_parallel.py:145). It excludes prefill and
queueing, whereas the aggregate is tokens divided by wall time (bench_parallel.py:144). So per-stream x np
is greater than the aggregate: 16.1 x 8 = 128.8 against 116.4 for qwen np=8.

**Fix:** label the column "decode-only" in RUN_APPROACH.md:35.

---

## Checked and fine

- **Threads, not asyncio.** urllib releases the GIL on I/O. The only locks are the slot free-list
  (run.py:440, 449-462) and the per-file writer lock (log.py:37-45). Neither is held across an HTTP call.
  The pure-Python prefix loop in `expected_new_tokens` (dialogue.py:68-84) and the per-call jinja compile
  are negligible against ~20 s turns.
- **The slot pool cannot underflow.** There are exactly `concurrency` threads and slots, and one job per
  thread. A dyad keeps its slot for life, and seeker and mentor use the same index on separate servers.
  `run` refuses a shared URL (run.py:392-396).
- **Every completion request carries `id_slot`:** dialogue.py:122, survey.py:97, run.py:139/144,
  scorer.py:287. No request goes through LCP/LRU slot selection.
- **`/tokenize` runs in llama-server's HTTP thread** (`b10488/server-context.cpp:4916-4925`), so the two
  tokenize calls per turn do not queue behind decode.
- **Within a process, lines never interleave.** All 288 rows in E3 and every E1 row parsed. Per-dyad order
  is always `started`, then turns and surveys, then `complete`/`failed`, and `complete` comes only after
  the post-survey (run.py:264-289).
- **Queued dyads never start after Ctrl-C, and the exit code is 130** (E5, single SIGINT). Queued dyads
  leave no status row. The window between SIGINT and `stop.set()` is microseconds.
- **Retries and attempts within one process** are deterministic from `status.jsonl`. There is one
  `(spec, attempt)` per dyad and no concurrent retries of the same dyad.
- **Surveys** run sequentially per dyad on the dyad's own mentor slot. Post items share the transcript
  prefix. They are about 1% of tokens and do not block other dyads.
- **`parallel_scaling.csv` matches the RUN_APPROACH.md table on every row:** aggregate, per-stream, cold
  prefill and GTT (rounded). The headline numbers in README.md and compute-budget.md match too. The Olmo
  np=16 row is in the doc but not the CSV, and it is marked crashed.
- **`bench_parallel.py` matches the harness setup where it matters.** It pins `id_slot`
  (bench_parallel.py:67), sends `cache_prompt: true`, and uses the same `-c = (depth+1024)*np` as
  RUN_APPROACH.md:16 and serve-*.sh. The context-budget check passes: 40 x 2 x 300 + 2048 = 26,048, within
  33,792 per slot.

## Reproduction

Everything is in `scratchpad/`:
- `fake_llama.py`: the fake server (`--slots`, `--wrap`, `--build`, `--reply-words`, `--decode-ms`).
- `servers.sh` and `stats.sh`: start the servers and read their stats.
- `hrun.py` and `hrun_grouped.py`: run the harness CLI, with the second one grouping scorer targets.
- `mkman.py` and `mkcfg.py`: write a dyad manifest and a config.
- `e5/drive.py`: the Ctrl-C driver.
- `e1`-`e7/`: each experiment's data and output.
