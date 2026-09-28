#!/bin/bash
# The study's three servers: seeker, mentor, judge. One llama-server per role, each at the operating
# point in models/RUN_APPROACH.md, each under a watchdog like the other serve-*.sh wrappers. This is
# the versioned launch script for the waves; the box's systemd tiers and the older wrappers are not.
#
# Operating point (RUN_APPROACH.md, "The operating point"), per server:
#   -np 8 slots, -c = (DEPTH + HEADROOM) * SLOTS = (32768 + 1024) * 8 = 270336, so each dialogue's
#   slot holds the full 32k (-c is divided across slots; see README.md in this directory);
#   q8_0 K and V cache; --cache-reuse 256; -ngl 999 -fa on.
#   --cache-ram 0 on EVERY role, not only Olmo (docs/audit/2026-09-28-parallelism.md L3): pinned
#   slots never read the host prompt cache back, it costs up to 8 GiB of unified memory per server
#   and a state copy per turn, and it is the code path that crashes on sliding-window models.
#   Exact-prefix reuse inside each pinned slot is unaffected.
#
# Chat templates: every role runs --jinja so /apply-template uses the GGUF's own template (the
# harness's template_parity check depends on it). The Olmo arm is the exception: b10488 rejects
# Olmo-3's template, so run that mentor with MENTOR_TEMPLATE_ARGS="--no-jinja --chat-template chatml"
# and expect `check` to warn server_chat_template while template_parity must still pass.
#
# Builds: b10488 (commit 9d77fa172) is the measured build. gpt-oss has no Vulkan MXFP4 path on this
# box and runs on the HIP build of the same commit:
#   MENTOR_LLAMA_BIN=/home/alex/.local/llamacpp/src/build-hip/bin
#
# Memory: three servers at this shape sit on top of the resident systemd tiers (~50 GiB). Budget
# against the ~75 GiB that is actually free (README.md) or stop the tiers first; np=8 for Olmo alone
# is 80 GiB with the tiers up. The judge runs after each wave, so it need not be up during one:
#   ./serve-study.sh start seeker mentor      # during a wave
#   ./serve-study.sh stop mentor; ./serve-study.sh start judge
#
# Model paths have no defaults: the seeker and the judge are chosen after the pilot
# (docs/decisions/factorial.md) and the mentor changes per arm. Everything is an environment
# variable; defaults are the box's layout:
#   SEEKER_GGUF=... MENTOR_GGUF=... JUDGE_GGUF=... ./serve-study.sh start
#
# The ports match harness/config.example.json for seeker (8201) and mentor (8202). Its judge url is
# :8099, which is the resident llm-deep tier, not a study server: point the config at JUDGE_PORT.
#
# Usage:  ./serve-study.sh start [seeker|mentor|judge ...]    (default: all three)
#         ./serve-study.sh stop  [seeker|mentor|judge ...]
# Each start appends the exact command line and `llama-server --version` to the role's log, so the
# log records what served the wave; the harness records the model sha and template per run.

set -u

LLAMA_BIN=${LLAMA_BIN:-/home/alex/.local/llamacpp/llama-b10488}   # build directory holding llama-server
LOG_DIR=${LOG_DIR:-/home/alex/llm-serving/study-logs}
HOST=${HOST:-127.0.0.1}      # loopback only; the harness runs on the same box
SLOTS=${SLOTS:-8}
DEPTH=${DEPTH:-32768}
HEADROOM=${HEADROOM:-1024}

SEEKER_PORT=${SEEKER_PORT:-8201}
MENTOR_PORT=${MENTOR_PORT:-8202}
JUDGE_PORT=${JUDGE_PORT:-8203}

# Per-role overrides; each falls back to the shared value above.
SEEKER_LLAMA_BIN=${SEEKER_LLAMA_BIN:-$LLAMA_BIN}
MENTOR_LLAMA_BIN=${MENTOR_LLAMA_BIN:-$LLAMA_BIN}
JUDGE_LLAMA_BIN=${JUDGE_LLAMA_BIN:-$LLAMA_BIN}
SEEKER_SLOTS=${SEEKER_SLOTS:-$SLOTS}
MENTOR_SLOTS=${MENTOR_SLOTS:-$SLOTS}
JUDGE_SLOTS=${JUDGE_SLOTS:-$SLOTS}
SEEKER_TEMPLATE_ARGS=${SEEKER_TEMPLATE_ARGS:---jinja}
MENTOR_TEMPLATE_ARGS=${MENTOR_TEMPLATE_ARGS:---jinja}
JUDGE_TEMPLATE_ARGS=${JUDGE_TEMPLATE_ARGS:---jinja}
# --alias: the harness reads the model family from the GGUF path, then the alias. An ollama blob path
# (sha256-...) names no family, so give such a model an alias that does, e.g. qwen3.6-35b-a3b.
SEEKER_ALIAS=${SEEKER_ALIAS:-}
MENTOR_ALIAS=${MENTOR_ALIAS:-}
JUDGE_ALIAS=${JUDGE_ALIAS:-}

ROLES="seeker mentor judge"

alive () {   # a live process; an exited child not yet reaped (state Z) does not count
  [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null && [ "$(ps -o stat= -p "$1" 2>/dev/null | cut -c1)" != Z ]
}

role_var () {   # role_var seeker PORT -> value of SEEKER_PORT
  local name
  name="$(echo "$1" | tr '[:lower:]' '[:upper:]')_$2"
  printf '%s' "${!name:-}"
}

# The watchdog for one role, run in the foreground by `start` under nohup. Relaunches llama-server
# if it dies (driver wedge insurance, as in serve-bulk.sh). A relaunch empties every slot's KV
# cache; the harness then fails the affected dyads with CacheReuseLost and retries them.
watch_role () {
  local role=$1 bin gguf port slots tmpl alias ctx srv rc
  bin=$(role_var "$role" LLAMA_BIN)
  gguf=$(role_var "$role" GGUF)
  port=$(role_var "$role" PORT)
  slots=$(role_var "$role" SLOTS)
  tmpl=$(role_var "$role" TEMPLATE_ARGS)
  alias=$(role_var "$role" ALIAS)
  [ -n "$alias" ] || alias=$(basename "$gguf" .gguf)
  ctx=$(( (DEPTH + HEADROOM) * slots ))
  local -a tmpl_args
  read -r -a tmpl_args <<< "$tmpl"
  local -a cmd=("$bin/llama-server" -m "$gguf" --alias "$alias" -ngl 999 -fa on
                -np "$slots" -c "$ctx" -ctk q8_0 -ctv q8_0 --cache-reuse 256 --cache-ram 0
                "${tmpl_args[@]}" --host "$HOST" --port "$port")
  srv=""
  trap '[ -n "$srv" ] && kill "$srv" 2>/dev/null; exit 0' TERM INT
  while true; do
    echo "$(date -Is) $role: $("$bin/llama-server" --version 2>&1 | tr '\n' ' ')"
    echo "$(date -Is) $role: ${cmd[*]}"
    LD_LIBRARY_PATH="$bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "${cmd[@]}" &
    srv=$!
    echo "$srv" > "$LOG_DIR/$role.server.pid"
    wait "$srv"
    rc=$?     # before any $(...): a command substitution resets $?
    echo "$(date -Is) $role server exited rc=$rc -- restart in 10s"
    sleep 10
  done
}

start_role () {
  local role=$1 gguf bin
  gguf=$(role_var "$role" GGUF)
  bin=$(role_var "$role" LLAMA_BIN)
  if [ -z "$gguf" ]; then
    echo "$role: set $(echo "$role" | tr '[:lower:]' '[:upper:]')_GGUF to the model file"; return 1
  fi
  [ -r "$gguf" ] || { echo "$role: cannot read $gguf"; return 1; }
  [ -x "$bin/llama-server" ] || { echo "$role: no llama-server in $bin"; return 1; }
  if alive "$(cat "$LOG_DIR/$role.watchdog.pid" 2>/dev/null)"; then
    echo "$role: already running (watchdog pid $(cat "$LOG_DIR/$role.watchdog.pid"))"; return 1
  fi
  echo "starting $role on $HOST:$(role_var "$role" PORT)  log: $LOG_DIR/llama-study-$role.log"
  nohup bash "$0" _watch "$role" >> "$LOG_DIR/llama-study-$role.log" 2>&1 &
  echo $! > "$LOG_DIR/$role.watchdog.pid"
}

stop_role () {
  local role=$1 pid spid
  pid=$(cat "$LOG_DIR/$role.watchdog.pid" 2>/dev/null)
  spid=$(cat "$LOG_DIR/$role.server.pid" 2>/dev/null)
  if alive "$pid"; then
    kill "$pid"; echo "stopped $role (watchdog pid $pid)"   # its TERM trap stops the server
    sleep 1
  else
    echo "$role: watchdog not running"
  fi
  # In case the watchdog died without its trap (kill -9), stop an orphaned server as well.
  if alive "$spid"; then
    kill "$spid" && echo "stopped orphaned $role server (pid $spid)"
  fi
  rm -f "$LOG_DIR/$role.watchdog.pid" "$LOG_DIR/$role.server.pid"
}

check_roles () {
  local r
  for r in "$@"; do
    case " $ROLES " in *" $r "*) ;; *) echo "unknown role: $r (expected: $ROLES)"; exit 1 ;; esac
  done
}

mkdir -p "$LOG_DIR"
action=${1:-}
[ $# -gt 0 ] && shift
[ $# -gt 0 ] || set -- $ROLES

case "$action" in
  start) check_roles "$@"; rc=0; for r in "$@"; do start_role "$r" || rc=1; done; exit $rc ;;
  stop)  check_roles "$@"; for r in "$@"; do stop_role "$r"; done ;;
  _watch) check_roles "$1"; watch_role "$1" ;;
  *) echo "usage: $0 start|stop [seeker|mentor|judge ...]"; exit 1 ;;
esac
