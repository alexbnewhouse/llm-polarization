#!/usr/bin/env python3
"""
Parallel-slot scaling benchmark for the LLM polarization study.

Measures what the study actually does: N concurrent dialogues, each holding a
deep KV cache, each generating one turn at a time.

Two regimes per configuration:
  COLD  first request per slot - full prefill of the depth-D context
  WARM  follow-up with cache_prompt=true - only the new turn is prefilled,
        which is the steady state once a dialogue is under way

WARM aggregate throughput is the number that decides how long a wave takes.

Usage (on the Framework Desktop / Strix Halo box):

  python3 bench_parallel.py \
      --model ~/llm-serving/gguf/Olmo-3-7B-Instruct-Q4_K_M.gguf \
      --label olmo3-7b-instruct-q4km \
      --plan  32768:1:q8_0,32768:4:q8_0,32768:8:q8_0,32768:16:q8_0 \
      --out   results/olmo3_7b_parallel.jsonl

Each --plan entry is depth:n_parallel:kv_type. The server is started fresh
for every entry with -c = (depth + 1024) * n_parallel and killed afterwards,
so nothing is left running. One JSON record per entry is appended to --out;
a hang or crash loses only the current entry.

History: the 2026-08-25 run that set the operating point (qwen3.6:35b-a3b,
see results/parallel_results.jsonl) used this same logic with the model and
plan hard-coded. The 2026-09-08 re-measurement of Olmo-3-7B added the
command-line interface; the measurement itself is unchanged.
"""
import argparse, json, os, subprocess, sys, threading, time, urllib.request

FILLER = ("The delegate reviewed the proposal, noted objections from the "
          "committee, and asked that the record reflect a formal dissent. ")
GTT = "/sys/class/drm/card1/device/mem_info_gtt_used"


def prompt_of(tokens, seed):
    """~4.6 chars/token for this filler; unique prefix keeps slots distinct."""
    need = int(tokens * 4.6)
    body = (FILLER * (need // len(FILLER) + 2))[:need]
    return f"[transcript {seed}] " + body


def wait_ready(proc, port, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(2)
    return False


def call(port, prompt, n_predict, cache):
    body = json.dumps({"prompt": prompt, "n_predict": n_predict,
                       "temperature": 0.7, "cache_prompt": cache}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.load(r)
    t = d.get("timings", {})
    return {"wall": time.time() - t0,
            "pp_n": t.get("prompt_n"), "pp_ts": t.get("prompt_per_second"),
            "tg_n": t.get("predicted_n"), "tg_ts": t.get("predicted_per_second")}


def run_round(port, prompts, n_predict, cache):
    """Fire all prompts concurrently; return per-stream results + wall time."""
    res = [None] * len(prompts)

    def work(i):
        try:
            res[i] = call(port, prompts[i], n_predict, cache)
        except Exception as e:                       # noqa: BLE001
            res[i] = {"error": str(e)[:120]}

    ths = [threading.Thread(target=work, args=(i,)) for i in range(len(prompts))]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return res, time.time() - t0


def run_config(args, depth, npar, kv):
    ctx = (depth + 1024) * npar
    cmd = [args.lcpp, "-m", args.model, "--host", "127.0.0.1", "--port", str(args.port),
           "-ngl", "999", "-fa", "on", "-np", str(npar), "-c", str(ctx),
           "-ctk", kv, "-ctv", kv, "--cache-reuse", "256",
           # llama.cpp >= b10488 parses the chat template at startup even though
           # this benchmark only uses /completion. Olmo-3's template uses a
           # filter the parser rejects, so disable it; throughput is unaffected.
           "--no-jinja"] + args.extra
    print(f"\n### depth={depth} np={npar} kv={kv} ctx={ctx}", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rec = {"label": args.label, "model": os.path.basename(args.model),
           "llama_build": os.path.basename(os.path.dirname(args.lcpp)),
           "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "depth": depth, "n_parallel": npar, "kv": kv, "ctx_total": ctx, "gen": args.gen}
    try:
        t_load = time.time()
        if not wait_ready(proc, args.port):
            rec["error"] = "server failed to start"
            print("   !! server failed to start", flush=True)
            return rec
        rec["load_s"] = round(time.time() - t_load, 1)

        prompts = [prompt_of(depth, i) for i in range(npar)]

        cold, cold_wall = run_round(args.port, prompts, 8, True)      # fills each slot
        warm, warm_wall = run_round(args.port,
                                    [p + "\n\nContinue the discussion." for p in prompts],
                                    args.gen, True)

        ok = [w for w in warm if "error" not in w and w.get("tg_n")]
        rec["streams_ok"] = len(ok)
        rec["cold_wall_s"] = round(cold_wall, 1)
        rec["cold_pp_tokens"] = sum(c.get("pp_n") or 0 for c in cold if "error" not in c)
        rec["cold_agg_prefill_tps"] = round(rec["cold_pp_tokens"] / cold_wall, 1) if cold_wall else None
        rec["warm_wall_s"] = round(warm_wall, 2)
        rec["warm_gen_tokens"] = sum(w["tg_n"] for w in ok)
        rec["warm_agg_tps"] = round(rec["warm_gen_tokens"] / warm_wall, 2) if warm_wall else None
        rec["warm_per_stream_tps"] = round(sum(w["tg_ts"] for w in ok) / len(ok), 2) if ok else None
        rec["warm_prefill_per_turn"] = round(sum((w.get("pp_n") or 0) for w in ok) / len(ok), 1) if ok else None
        rec["gtt_gib"] = round(int(open(GTT).read()) / 1073741824, 1) if os.path.exists(GTT) else None
        errs = [w["error"] for w in warm if "error" in w]
        if errs:
            rec["stream_errors"] = errs[:3]
        print(f"   agg={rec['warm_agg_tps']} tok/s  per-stream={rec['warm_per_stream_tps']}"
              f"  gtt={rec['gtt_gib']}GiB  ok={rec['streams_ok']}/{npar}", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        time.sleep(5)                                # let GTT settle before the next config
    return rec


def parse_plan(s):
    plan = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        depth, npar, kv = item.split(":")
        plan.append((int(depth), int(npar), kv))
    return plan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path to the GGUF")
    ap.add_argument("--label", required=True, help="short study label written into every record")
    ap.add_argument("--out", required=True, help="JSONL file to append records to")
    ap.add_argument("--plan", default="32768:1:q8_0,32768:4:q8_0,32768:8:q8_0,32768:16:q8_0",
                    help="comma-separated depth:n_parallel:kv entries")
    ap.add_argument("--lcpp", default=os.environ.get("LCPP", os.path.expanduser(
                    "~/.local/llamacpp/llama-b10488/llama-server")), help="llama-server binary")
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--gen", type=int, default=200, help="tokens per turn; 200 matches the study design")
    ap.add_argument("--extra", default="", help="extra llama-server flags, space-separated")
    args = ap.parse_args()
    args.model = os.path.expanduser(args.model)
    args.extra = args.extra.split()
    plan = parse_plan(args.plan)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f"model={args.model}\nlabel={args.label}\nllama-server={args.lcpp}\nplan={plan}", flush=True)
    with open(args.out, "a") as f:
        for depth, npar, kv in plan:
            rec = run_config(args, depth, npar, kv)
            f.write(json.dumps(rec) + "\n")
            f.flush()
    print("\nPARALLEL BENCH COMPLETE", flush=True)


if __name__ == "__main__":
    main()
