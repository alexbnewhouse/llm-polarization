# Dyad harness

`harness/` runs seeker/mentor dialogues against two llama-server endpoints, administers the mentor's
pre/post survey batteries, and scores seeker adherence offline. Design: `docs/superpowers/specs/2026-09-08-dyad-harness-design.md`.

## Use

```bash
pip install -r harness/requirements.txt            # jinja2 only; gguf-py comes from the llama.cpp checkout
cp harness/config.example.json config.json         # edit urls, gguf_py_path, run_seed
python -m harness.run check  --config config.json
python -m harness.run run    --config config.json --manifest dyads.jsonl --run-id pilot-2026-09-18
python -m harness.run score  --config config.json --run-id pilot-2026-09-18 --scope pilot
python -m harness.run survey --config config.json --run-id pilot-2026-09-18 --phase post   # re-administer
python -m pytest harness/tests -q                  # unit tests; HARNESS_LIVE_URL=... adds the live test
```

Servers are started outside the harness with the flags in `models/RUN_APPROACH.md`. Output lands in
`data/<run_id>/` as `manifest.json`, `dyads.jsonl`, `status.jsonl`, `turns.jsonl`, `surveys.jsonl`, `scores.jsonl`.

## Terms (fixed 2026-09-02)

- **Seeker**: the persona-prompted LLM standing in for a human who comes to the
  conversation for guidance.
- **Mentor**: the zero-shot LLM advisor whose responses are the outcome.
- **Dyad**: one seeker paired with one mentor for one dialogue.
- Chat-template roles (system, user, assistant) are a separate axis: each agent
  sees its own lines as assistant-role messages and the other's as user-role
  messages.

## Four things the harness must do

1. **Egocentric context projection** (`luo2026spasm`). Keep one canonical
   transcript. Before each generation, build that agent's message list with
   its own lines as assistant-role messages and the other agent's lines as
   user-role messages. Never pass a shared transcript with speaker labels in
   the text. This is what stops echoing and role confusion in LLM-to-LLM
   dialogue.
2. **A persona-mode switch**, once or reinforced, logged per dialogue. In
   reinforced mode, append the compact reminder from the seeker template as the
   final message before the seeker generates. Put it after the history, not
   before, so the cached prefix (system prompt plus history) stays valid.
3. **Nothing on the mentor side** beyond what the chat template requires. Log
   the exact template string and the model hash. Any system text on the mentor
   is treatment.
4. **Per-turn log row**: dialogue id, turn index, agent (seeker or mentor),
   model hash, persona mode, prompt token count, generation, and an empty
   adherence column the scorer fills in later.

## Serving contract the harness talks to

llama.cpp `llama-server` on the Framework Desktop, one process per model,
`-np 8` slots, q8_0 KV cache, `-c = (depth + 1024) * 8`. Each dialogue keeps
its own slot so every turn only prefills the new tokens. See
`models/RUN_APPROACH.md` for the operating point and the throughput each arm
achieves there. The single largest lever is KV cache reuse: a turn at 32k
depth should prefill tens of tokens, not 32,768. Log `prompt_n` from the
server's `timings` on every call so a lost cache is visible immediately.

## Persona-stability requirements the pilot decides

- Default to **reinforced** delivery (compact reminder every seeker turn).
- W4 pilot: both delivery modes, 40 turns, at least five dyads each, adherence
  scored every turn. If stated-once keeps mean adherence at or above 0.8
  through turn 40, switch to stated-once; otherwise keep reinforced.
- Never reinforce the mentor.
- Flag dialogues where seeker adherence falls under threshold for three
  consecutive turns; decide exclusion vs compliance-weighting in the
  pre-analysis plan.
- Fix one seeker model across every arm; choose it by measured adherence, not
  size; use a different model family from the mentor.

Full reasoning and citations: `docs/design/persona-stability.md`.
