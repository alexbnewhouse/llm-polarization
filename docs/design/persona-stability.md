# Persona stability

Research pass done 2026-09-02. Exported from the Notion page "Persona
stability" under the project hub. Every cited key is in
`paper/references.bib`.

> **Superseded in part, 2026-09-10.** The four questions this pass fed are now
> decided in `docs/decisions/persona-stability.md`. Three recommendations below
> were amended in the decision: the pilot A/B became a measurement rather than a
> gate on delivery mode; the 0.8 threshold is calibrated on hand labels rather
> than borrowed from `li2024instability`; and the low-adherence rule is settled
> as keep-under-ITT rather than "either is defensible." The evidence synthesis
> and the lever table are unchanged.

**Bottom line.** Persona drift in LLM-to-LLM dialogue is real, shows up within
eight turns, and is not fixed by using a bigger model. Stability is something
the harness has to manufacture and the pipeline has to measure, not something
the persona prompt delivers on its own. Four moves: build the harness so each
agent sees the dialogue from its own side; put a rich narrative persona in the
seeker's system prompt and re-inject a compact reminder before each seeker
turn; score both agents every turn with an automatic adherence judge; let the
40-turn pilot decide whether the reminder can be dropped.

## The three questions

1. How do we maintain stability in the seeker's persona?
2. Does the choice of seeker model matter?
3. Does the choice of mentor model matter?

The original two unformed thoughts were: keep persona guidelines in context
at every turn, and use a very large model as the seeker. The first is right
but needs sharpening (the persona is already in context every turn; the
problem is attention, not presence). The second is contradicted by the
evidence and should be dropped as a stability strategy.

## 1. What the evidence says about drift

### It happens within a handful of turns

- `li2024instability` is the closest published analogue. Two system-prompted
  chatbots self-chat for eight rounds. On LLaMA-2-70B, adherence falls from
  about 95% at round one to about 75% at round eight. The mechanism is
  attention decay: the system prompt never leaves the context; the model just
  stops looking at it.
- `dongre2026attention`: goal-defining tokens lose attention accessibility
  across turns even when still decodable from the residual stream; failure
  timing is predictable per architecture.
- `laban2025lost`: across 200k simulated conversations, an average 39%
  performance drop multi-turn versus single-turn. Models commit early and
  fail to recover.
- `qin2024sysbench` names multi-turn instability of system-message following
  as one of three failure classes.
- `khetan2026politicsbench` shows the opposite direction is also possible:
  stance commitment rises about 1.4 points on a 0-to-5 scale from the initial
  to the decision stage. Drift is not only decay toward neutral; it can be
  intensification. `nudo2025exaggeration` calls this generative exaggeration.

### Bigger models do not fix it

- `choi2024identity`, nine models: larger models drift more, family matters
  less than parameter count, and assigning a persona does not by itself keep
  identity stable.
- `tosato2025persistent`: even 400B-plus models show SD above 0.3 on 5-point
  scales; conversation history can increase variability.
- `bernardelle2025shifts`: 70B-plus models have a more reliable ideological
  range than 7-to-8B models. Size buys steerability, not stability.
- Implication: pick the seeker model by measured adherence in the pilot, not
  by parameter count. API models break the local, reproducible, hash-pinned
  setup.

### LLM-to-LLM dyads have their own pathologies

- **Echoing and role confusion.** `luo2026spasm` documents agents parroting
  each other and slipping into the other role. Their fix, Egocentric Context
  Projection, keeps one canonical transcript and projects it into each
  agent's own perspective before generation. This removed echoing entirely.
- **Attractor states.** `ko2026attractor`: each model has its own
  conversational attractor in self-play; in mixed-model debates influence is
  asymmetric.
- **Conformity.** `hao2026flips` decompose stance flips: 37% spontaneous
  instability, 29% harmful conformity, the rest persuasion. This is the
  vocabulary the results section will need.
- **Sycophancy under sustained pressure.** `nogueira2026persuasion`: five
  turns of escalating argument raise sycophantic alignment from a median 50%
  to 79%. `jain2026interaction`: multi-turn context and accumulated user
  profiles raise agreement sycophancy sharply. `kelley2026personalization`: a
  model framed as a peer abandons positions under personalized challenge far
  more than one framed as an advisor.
- **Persona collapse.** `xiao2026chameleon`: agents given distinct profiles
  converge to a narrow behavioral mode. Within-cell variance is a thing to
  watch, not only cell means.

## 2. Levers, ranked

| Lever | What it does | Evidence | Cost | Verdict |
|---|---|---|---|---|
| Egocentric context projection | Each agent receives the history with its own lines as assistant-role and the partner's as user-role. No shared labelled transcript. | `luo2026spasm` | Free; it is just building the harness correctly | **Required** |
| Rich narrative persona in the seeker's system prompt | Backstory, values, two or three explicit stance anchors in the persona's own words | `li2026spirit`; `hu2024persona` | Free | **Required** |
| Compact per-turn reminder | Append a two-to-three sentence persona summary as the last thing the seeker sees before it generates | `li2024instability` | About 100 tokens per turn; 40 turns is 4k tokens inside a 32k window | **Recommended default; pilot decides** |
| Per-turn adherence scoring | A judge model scores each seeker turn on prompt-to-line, line-to-line, and Q&A consistency; mentor stance scored the same way | `abdulhai2025consistently`; `li2024instability` | One judge call per scored turn, offline on logs | **Required as manipulation check** |
| Persona vectors / activation monitoring | Project activations onto a trait direction each turn | `chen2025persona`; `lu2026assistant`; `vogel2024controlvectors` | Weeks of engineering | Not for this paper; limitations and future work |
| Split-softmax or CFG decoding | Reweight attention or logits toward the system prompt | `li2024instability` | Custom decoding loop, not in the llama.cpp server path | No |
| RL fine-tuning for consistency | Train the seeker with consistency rewards | `abdulhai2025consistently` | A training pipeline | No |

## 3. Recommendation

### Q1. Maintaining the seeker's persona

- **Persona delivery is a manipulation, not an object of study.** A decaying
  seeker persona is a decaying dose, and insufficient dose is the main null
  risk. Hold the dose constant and measure that you did.
- **Default to reinforced delivery.** The seeker's system prompt carries the
  full narrative persona. Before each seeker generation the harness appends a
  compact reminder: name, ideology, the stance anchors, the openness level,
  phrased as a note to self.
- **Pre-registrable decision rule.** In the W4 pilot, run both delivery modes
  at 40 turns with at least five dyads each. Score adherence every turn. If
  the stated-once arm keeps mean adherence at or above 0.8 through turn 40,
  switch to stated-once for naturalism. Otherwise keep reinforced.
- **Never reinforce the mentor.** Anything injected on that side is treatment.
- **Write the low-adherence rule now.** Dialogues where seeker adherence
  falls under threshold for three consecutive turns get flagged. Decide in
  the pre-analysis plan whether they are excluded or kept with a
  compliance-weighted estimate.

### Q2. Does the choice of seeker model matter

Yes, on four counts.

- **Fix one seeker model across every arm.** If the seeker varies with the
  mentor, persona quality becomes a confound on the treatment.
- **Choose it by measured adherence, not size.** Same five-dyad, 40-turn
  adherence check on each candidate; take the highest mean adherence at turn
  40, tie-break on tokens per second. *Amended 2026-09-11
  (`docs/decisions/factorial.md`): throughput is a compute criterion, not only
  a tie-break. The seeker generates half of every arm's tokens, so its tok/s
  sets roughly 8 to 28 days of the wave budget across the plausible range.*
- **Use a different model family from the mentor.** Same-model dyads converge
  to that model's attractor.
- **Expect asymmetric dose between the two ideological arms**
  (`bernardelle2025mapping`, `bernardelle2025shifts`). Report adherence
  separately by ideology arm.

### Q3. Does the choice of mentor model matter

Yes, and it is already the model arm (`docs/decisions/model-arm.md`).

- **The mentor's role framing is a design parameter with a known effect**
  (`kelley2026personalization`). The name mentor commits to the advisor
  frame; because the mentor is zero-shot, that frame comes from the seeker's
  side. Write it into the seeker template and hold it constant.
- **Sycophancy scales with exactly the dose this design increases.** The
  no-persona control arm separates mentor drift from sycophantic mirroring.
  Use the `hao2026flips` decomposition as the reporting language.

## 4. Judge cost, the one new constraint

Scoring both agents on every turn of every dialogue is 9,300 x 40 x 2 =
744,000 judge calls. Options, cheapest first:

- Score only the seeker in main runs (the mentor's stance is already captured
  by the pre and post survey); score both in the pilot. Halves it.
- Score every fourth turn plus the final turn in main runs. Quarters it again.
- Run the judge on the smaller local model, batched, after each wave
  finishes, so it never competes with dialogue generation for the box.
- The judge must not be the seeker or the mentor of that dialogue.

## 5. Still open

- Probe questions (`li2024instability`) versus judge-on-transcript
  (`abdulhai2025consistently`). Probes perturb the dialogue, so use them only
  in the pilot.
- Whether to keep a small stated-once condition in the main grid so persona
  decay itself is reportable. Only if the pilot shows a large gap.
- The 0.8 threshold is borrowed; calibrate on pilot data before
  pre-registering.
