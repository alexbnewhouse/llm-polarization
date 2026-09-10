# Research design

Exported from the Notion page "LLM Political Polarization Research - Project
Hub" (last edited 2026-09-02). The Notion hub remains the live task tracker;
this file is the design of record for the repository.

**North star:** establish LLM-to-LLM dialogue as an experimental laboratory
for measuring when and how chatbots produce polarized political responses.

**Research question:** how does the ideological slant of a user affect the
political polarization of LLM responses after long dialogues?

**Deadline:** full draft by 2026-11-30. Funded by IHS (USD 3,000, which bought
the Framework Desktop with 128 GB unified memory that runs every dialogue).

## Terminology (fixed 2026-09-02)

- **Seeker** = the persona-prompted LLM standing in for a human who comes to
  the conversation for guidance.
- **Mentor** = the zero-shot LLM advisor whose responses are the outcome.
- **Dyad** = one seeker paired with one mentor for one dialogue.
- Chat-template roles (system, user, assistant) are a separate axis: each
  agent sees its own lines as assistant-role messages and the other's as
  user-role messages. Older notes that say "user LLM" or "human stand-in"
  mean seeker; "responder" means mentor.

## The design

Multiple instances of the same model (or different models) hold structured
conversations in dyads. The seeker plays a human with an assigned political
persona and comes to the conversation looking for guidance. The mentor is the
LLM advisor; it starts zero-shot with no prompting, and its movement is what
gets measured.

Starting prompt attributes for the seeker are varied systematically and
treated as experimental conditions: subject matter, geographic and social
role, and degree of openness or reservation in stated preferences. Repeated
trials per condition permit causal estimation of whether particular framings
increase polarized output from the mentor, and separate stochastic
variability from systematic bias.

**Issues for discussion:** militarization of immigration enforcement;
decarbonization of the economy.

**Open design questions** from the 2025-11-04 column meeting, scheduled in W1.
All three are now closed, which unblocks the factorial freeze:

| Question | Decided | Record |
|---|---|---|
| Persona stability over time | 2026-09-10 | `docs/decisions/persona-stability.md` |
| Is the seeker always the same LLM, or varied? | 2026-09-08 | Fixed seeker, chosen by measured adherence, different family from the mentor (`docs/design/persona-stability.md` Q2) |
| How to vary LLMs across architectures | 2026-09-08 | `docs/decisions/model-arm.md` |

## Survey instruments

See `instruments/survey-batteries.md`. Key reference: de Jong (2024),
Communications Psychology 2(1):5.

## Scale

Roughly 9,300 dialogues across the pilot, the baseline, and three run waves.
Roughly 16,750 words (a 13,000-word paper plus pre-analysis plan and
reproducibility appendix), written 2026-10-05 to 2026-11-30.

**The dialogue count needs recomputing.** That 9,300 assumed four arms at 20
turns. The arm is now three at 40 turns (`docs/decisions/persona-stability.md`),
which is a different N and a different per-cell count. Set it in the factorial
freeze from the measured budget in `models/RUN_APPROACH.md`, not from this
number.

The two projects (this and the prospectus due 2026-09-30) consume different
resources: the prospectus needs attention, the experiments need electricity.
Weeks 1 through 6 are built so this project's demand on the researcher stays
near zero while the machine does the expensive part.

## Where things stand (2026-09-10)

| Piece | State |
|---|---|
| Research question and framing | Written, funded, defended in the IHS application |
| Survey instruments | de Jong (2024) identified, not yet adapted to the US context |
| The factorial | **Unblocked 2026-09-10.** All three W1 design questions are answered: seeker fixed-or-varied (2026-09-08), model arm (2026-09-08), persona stability (2026-09-10) |
| Hardware | Online, tuned, benchmarked. 128 GiB GTT confirmed, operating point measured |
| Model arm | Decided 2026-09-08 (`docs/decisions/model-arm.md`). **Three arms, not four**: choosing 40 turns cuts glm-4.7-flash |
| Compute budget | All four arms measured 2026-09-08: about 19.8 days at 20 turns, 39.5 at 40, 22.3 for three arms at 40. **40 turns is the chosen row** |
| Pipeline | Built. `harness/`: dialogue engine, surveys, offline adherence scorer, and a CLI with `check`, `run`, `survey`, `score`. 94 unit tests pass, 2 live tests skip without a server. Pre-pilot gate: `harness check` against the real Olmo server |
| Data | None |

The surplus from tuning (about 17 days) should go to design, not to N: the
threats to this paper are a sycophancy confound and a possible
insufficient-dose null, neither of which more trials can fix. Spend it on
40-turn dialogues and a no-persona control arm. Both are now decided: 40 turns
as of 2026-09-10, at the cost of the fourth model arm. If further surplus
appears, `docs/decisions/persona-stability.md` argues for stated-once cells as
a dose-response contrast ahead of restoring that arm.

## The runway

| Week | Focus | Prospectus load |
|---|---|---|
| W1 Aug 24-30 | Box online; answer the three open design questions | Heavy |
| W2 Aug 31-Sep 6 | Freeze the factorial; adapt de Jong to US context | Heavy |
| W3 Sep 7-13 | Dyad harness, logging, randomizer | Heavy |
| W4 Sep 14-20 | Pilot run (moved to Sep 18); read transcripts; validate indices | Heavy |
| W5 Sep 21-27 | Pre-register; baseline run executes | Peak, machine-heavy on purpose |
| W6 Sep 28-Oct 4 | Machine only: verify, snapshot, stage wave 1 | Prospectus due Sep 30 |
| W7 Oct 5-11 | Wave 1 runs; draft intro and theory | Done |
| W8 Oct 12-18 | Wave 2 runs; analysis code and lit review | Done |
| W9 Oct 19-25 | Checkpoint Oct 19; wave 3 runs; first ATEs | Done |
| W10 Oct 26-Nov 1 | Full analysis, heterogeneity, all figures | Done |
| W11 Nov 2-8 | Write methods and results | Done |
| W12 Nov 9-15 | Revise intro and theory; discussion; limitations | Done |
| W13 Nov 16-22 | Conclusion, assemble, read-through, abstract | Done |
| W14 Nov 23-30 | Buffer: repro appendix, bib, format, send | Done |

**The Oct 19 checkpoint is a pre-committed descope.** The cut list was: drop
the third model arm, then collapse openness to two levels, then drop a topic.
**The first item is already spent** — glm-4.7-flash was cut on 2026-09-10 to
pay for 40-turn dialogues — so the checkpoint now starts at collapsing openness
to two levels. Never cut Olmo (see `docs/decisions/model-arm.md`).

## Technical notes

Model configuration:

- Separate baseline LLMs from experimental LLMs to avoid leakage.
- Seekers receive hidden persona prompts; mentors start zero-shot.
- Record architecture, quantization, and revision hash for every model used.

Data collection:

- Log all dialogue turns with run_id, dyad_id, turn index, condition, model hash.
- Track persona and topic assignments.
- Store pre- and post-treatment survey responses.

Key challenges:

- Computational resources: the 128 GB box is the binding constraint on N.
  Set N from the compute budget (`models/RUN_APPROACH.md`), not from hope.
- Memory limitations: context window constraints may degrade effects over
  very long interactions. Context strategy is an explicit W4 decision.
- Validity trade-offs: may not capture longer-term change across multiple
  conversations, but provides direct measurement of current state.
- Ethical advantage: avoids exposing human participants to potentially
  polarizing interactions.

## Target venues

APSR, AJPS, ICWSM, ACM.
