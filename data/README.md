# data/

Experiment output lives here on the box that produced it and is **git-ignored**
(see `.gitignore`). Nothing in this directory is committed except this file.

Planned layout (from the logging-schema task; adjust when the harness exists):

```
data/
  pilot/                 W4 pilot: 5 dyads x condition x delivery mode, 40 turns
  baseline/              W5 baseline run
  wave1/ wave2/ wave3/   W7 to W9 runs, one directory per wave
  <run>/dialogues.jsonl  one row per turn: run_id, dyad_id, turn, agent, model_hash,
                         persona_mode, prompt_n, generation, adherence (filled later)
  <run>/surveys.jsonl    pre/post batteries per dialogue
  <run>/manifest.json    model files + sha256, llama.cpp build, server flags, seed,
                         condition grid, git commit of this repo
```

Archive completed runs to the NAS or Dropbox; record where in the manifest.
