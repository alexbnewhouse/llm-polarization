#!/usr/bin/env python3
"""Resolve the readable local model set to (label, path, timeout) lines."""
import glob, json, os, sys

ROOT = "/home/alex/.ollama/models"
GGUF = "/home/alex/llm-serving/gguf"

# ollama tag -> (study label, timeout seconds)
WANT = {
    "gpt-oss:20b":                 ("gpt-oss:20b (MoE 3.6B act)",        1200),
    "gpt-oss:120b":                ("gpt-oss:120b (MoE 5.1B act)",       2400),
    "qwen3.6:35b-a3b-mtp-q4_K_M":  ("qwen3.6:35b-a3b MTP (MoE 3B act)",  1200),
    "qwen3.8:27b":                 ("qwen3.8:27b (dense 27.8B)",         1800),
    "gemma3:27b":                  ("gemma3:27b (dense 27B)",            1800),
    "glm-4.7-flash:q4_K_M":        ("glm-4.7-flash (MoE)",               1200),
    "mixtral:8x7b":                ("mixtral:8x7b (MoE 12.9B act)",      1500),
    "mistral-small:latest":        ("mistral-small (dense 24B)",         1500),
    "llama3.1:8b":                 ("llama3.1:8b (dense 8B)",             900),
    "llama3.1:70b":                ("llama3.1:70b (dense 70B)",          3000),
}
STANDALONE = [
    ("Qwen3-4B-Instruct-2507 (dense 4B)",   f"{GGUF}/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",   900),
    ("Qwen3-30B-A3B-2507 (MoE 3B act)",     f"{GGUF}/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf", 1200),
]

found = {}
for mf in glob.glob(f"{ROOT}/manifests/**/*", recursive=True):
    if not os.path.isfile(mf):
        continue
    try:
        m = json.load(open(mf))
    except Exception:
        continue
    if "layers" not in m:
        continue
    parts = mf.split("/manifests/")[-1].split("/")
    tag = f"{'/'.join(parts[2:-1]) if len(parts) > 3 else parts[-2]}:{parts[-1]}"
    if tag not in WANT:
        continue
    for l in m["layers"]:
        if l.get("mediaType", "").endswith("model"):
            p = f"{ROOT}/blobs/{l['digest'].replace('sha256:', 'sha256-')}"
            if os.path.exists(p) and os.access(p, os.R_OK):
                found[tag] = p

rows = []
for label, p, tmo in STANDALONE:
    if os.access(p, os.R_OK):
        rows.append((label, p, tmo, os.path.getsize(p)))
    else:
        print(f"# unreadable: {p}", file=sys.stderr)
for tag, (label, tmo) in WANT.items():
    if tag in found:
        rows.append((label, found[tag], tmo, os.path.getsize(found[tag])))
    else:
        print(f"# not found/readable: {tag}", file=sys.stderr)

rows.sort(key=lambda r: r[3])          # smallest first: fast results early
for label, p, tmo, sz in rows:
    print(f"{label}|{p}|{tmo}")
    print(f"# {label:<36} {sz/1e9:6.1f} GB", file=sys.stderr)
