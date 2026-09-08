# prompts/

Seeker persona templates and the compact per-turn reminder go here once
written (Notion task: "Write the seeker persona prompt template"). Requirements
from `docs/design/persona-stability.md`:

- Rich narrative persona: backstory, values, two or three explicit stance
  anchors in the persona's own words, not a demographic list.
- The persona opens by asking for guidance, which fixes the mentor in the
  advisor frame. Hold that constant across every cell.
- A two-to-three sentence reminder (name, ideology, stance anchors, openness)
  phrased as a note to self, appended after the history in reinforced mode.
- Nothing is ever injected on the mentor side.

Version every template; the harness logs the template hash per dialogue.
