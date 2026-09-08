#!/usr/bin/env bash
# ROCm Verification Script for gfx1151 (Radeon 8060S / Strix Halo)
# Checks: driver, runtime, GPU detection, compute capability, Ollama GPU offload

set -uo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'
BOLD='\033[1m'; NC='\033[0m'

PASS=0; WARN=0; FAIL=0

pass()  { echo -e "${GREEN}[PASS]${NC} $*"; ((PASS++)); }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; ((FAIL++)); }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; ((WARN++)); }
info()  { echo -e "${CYAN}      ${NC} $*"; }
hdr()   { echo -e "\n${BOLD}── $* ──${NC}"; }

echo -e "${BOLD}ROCm Verification — AMD Ryzen AI MAX+ 395 / Radeon 8060S${NC}"
echo    "$(date)"
echo    "────────────────────────────────────────────────────────"

# ── 1. Kernel module ─────────────────────────────────────────────────────────
hdr "Kernel Modules"

if lsmod | grep -q amdgpu; then
    pass "amdgpu kernel module loaded"
else
    fail "amdgpu kernel module NOT loaded — run: sudo modprobe amdgpu"
fi

if lsmod | grep -qE "amd_iommu_v2|iommu"; then
    pass "IOMMU module loaded"
else
    warn "Could not confirm IOMMU via lsmod (check BIOS if issues arise)"
fi

# ── 2. ROCk / KFD ────────────────────────────────────────────────────────────
hdr "ROCk Driver (KFD)"

if [[ -e /dev/kfd ]]; then
    pass "/dev/kfd present"
    if [[ -r /dev/kfd ]]; then
        pass "/dev/kfd readable by current user"
    else
        fail "/dev/kfd not readable — add user to 'render' group: sudo usermod -aG render,video $USER"
    fi
else
    fail "/dev/kfd not found — ROCk driver may not be loaded"
fi

if [[ -e /dev/dri/renderD128 ]]; then
    pass "/dev/dri/renderD128 present"
else
    warn "/dev/dri/renderD128 not found — GPU may not be visible to ROCm"
fi

# ── 3. ROCm tools ────────────────────────────────────────────────────────────
hdr "ROCm Userspace Tools"

for tool in rocm-smi rocminfo; do
    if command -v "$tool" &>/dev/null; then
        ver=$("$tool" --version 2>/dev/null | head -1 || "$tool" 2>/dev/null | grep "Runtime Version" | head -1 || echo "installed")
        pass "$tool found ($ver)"
    else
        fail "$tool not found — install rocm package"
    fi
done

if command -v clinfo &>/dev/null; then
    pass "clinfo found (OpenCL available)"
else
    warn "clinfo not found (OpenCL may still work via ROCm)"
fi

# ── 4. GPU Detection ──────────────────────────────────────────────────────────
hdr "GPU Detection"

SMI_OUT=$(rocm-smi 2>/dev/null)
if echo "$SMI_OUT" | grep -q "gfx1151\|0x1586\|8060S\|STRXLGEN"; then
    pass "Radeon 8060S (gfx1151) detected by rocm-smi"
elif echo "$SMI_OUT" | grep -q "GPU\[0\]"; then
    pass "GPU detected by rocm-smi (verify it is gfx1151)"
else
    fail "GPU not detected by rocm-smi"
fi

ROCMINFO_OUT=$(rocminfo 2>/dev/null)
if echo "$ROCMINFO_OUT" | grep -q "gfx1151"; then
    pass "gfx1151 ISA confirmed by rocminfo"
    info "Target triple: amdgcn-amd-amdhsa--gfx1151"
else
    fail "gfx1151 not found in rocminfo — GPU may not be recognized"
fi

if echo "$ROCMINFO_OUT" | grep -q "KERNEL_DISPATCH"; then
    pass "GPU has KERNEL_DISPATCH capability (compute ready)"
else
    warn "KERNEL_DISPATCH not confirmed — compute workloads may fail"
fi

# ── 5. Memory ─────────────────────────────────────────────────────────────────
hdr "Unified Memory"

RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
if [[ $RAM_GB -ge 96 ]]; then
    pass "System RAM: ${RAM_GB}GB (excellent for large models via unified memory)"
elif [[ $RAM_GB -ge 32 ]]; then
    pass "System RAM: ${RAM_GB}GB (good)"
else
    warn "System RAM: ${RAM_GB}GB (consider 64GB+ for large models)"
fi

XNACK=$(rocminfo 2>/dev/null | grep "XNACK" | head -1)
if echo "$XNACK" | grep -qi "YES\|enabled"; then
    pass "XNACK enabled — optimal for unified memory access"
else
    warn "XNACK disabled — enable via: HSA_XNACK=1 or kernel parameter iommu.passthrough=0"
    info "For APUs with unified memory, XNACK=1 can improve GPU↔CPU memory sharing"
fi

# ── 6. OpenCL via ROCm ───────────────────────────────────────────────────────
hdr "OpenCL"

if command -v clinfo &>/dev/null; then
    CL_DEVICES=$(clinfo 2>/dev/null | grep "Device Name" | grep -v "^  Platform" | wc -l)
    if [[ $CL_DEVICES -gt 0 ]]; then
        pass "OpenCL sees $CL_DEVICES compute device(s)"
        clinfo 2>/dev/null | grep "Device Name" | grep -v "Platform" | while read -r line; do
            info "$line"
        done
    else
        fail "No OpenCL devices found"
    fi
fi

# ── 7. Ollama GPU offload ─────────────────────────────────────────────────────
hdr "Ollama GPU Offload"

if ! command -v ollama &>/dev/null; then
    warn "Ollama not installed — skipping"
else
    OLLAMA_VER=$(ollama --version 2>/dev/null)
    pass "Ollama installed: $OLLAMA_VER"

    # Check if ollama service is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        pass "Ollama API reachable at localhost:11434"

        # Find smallest available model for test
        TEST_MODEL=$(ollama list 2>/dev/null | awk 'NR>1 {print $1}' | head -1)
        if [[ -n "$TEST_MODEL" ]]; then
            info "Testing GPU offload with: $TEST_MODEL"

            # Get GPU use before
            GPU_BEFORE=$(rocm-smi --showuse 2>/dev/null | grep "GPU use" | awk '{print $NF}' | tr -d '%' || echo "0")

            # Run inference
            curl -s http://localhost:11434/api/generate \
                -H "Content-Type: application/json" \
                -d "{\"model\": \"$TEST_MODEL\", \"prompt\": \"Count to 5.\", \"stream\": false}" \
                > /tmp/ollama_test_$$ 2>&1 &
            CURL_PID=$!

            sleep 2
            GPU_DURING=$(rocm-smi --showuse 2>/dev/null | grep "GPU use" | awk '{print $NF}' | tr -d '%' || echo "0")
            wait $CURL_PID 2>/dev/null || true

            RESPONSE=$(cat /tmp/ollama_test_$$ 2>/dev/null)
            rm -f /tmp/ollama_test_$$

            if echo "$RESPONSE" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('response',''))" 2>/dev/null | grep -q "[0-9]"; then
                pass "Inference completed successfully"
            else
                warn "Inference response unexpected — check manually"
            fi

            if [[ $GPU_DURING -gt 2 ]]; then
                pass "GPU active during inference (${GPU_DURING}% utilization)"
            else
                warn "Low GPU utilization during inference (${GPU_DURING}%) — model may be CPU-bound or too small to measure"
            fi

            # Check processor assignment
            PS_OUT=$(ollama ps 2>/dev/null)
            if echo "$PS_OUT" | grep -q "GPU"; then
                pass "Ollama reports model using GPU"
                info "$PS_OUT"
            elif echo "$PS_OUT" | grep -q "CPU"; then
                fail "Ollama reports model using CPU only — check ROCm GPU detection"
            else
                info "No active model in ollama ps (model may have been unloaded)"
            fi
        else
            warn "No models installed — run: ollama pull llama3.1:8b"
        fi
    else
        warn "Ollama API not responding — start with: ollama serve"
    fi
fi

# ── 8. HSA / HIP environment ─────────────────────────────────────────────────
hdr "HSA / HIP Environment"

for var in HSA_OVERRIDE_GFX_VERSION ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES GPU_DEVICE_ORDINAL HSA_XNACK; do
    val="${!var:-}"
    if [[ -n "$val" ]]; then
        info "$var=$val (set)"
    fi
done

ROCM_PATH_SEARCH=""
for p in /opt/rocm /usr /usr/local; do
    if command -v "$p/bin/rocm-smi" &>/dev/null 2>&1 || [[ -f "$p/bin/rocm-smi" ]]; then
        ROCM_PATH_SEARCH="$p"; break
    fi
done
if [[ -z "$ROCM_PATH_SEARCH" ]] && command -v rocm-smi &>/dev/null; then
    ROCM_PATH_SEARCH="system PATH"
fi
if [[ -n "$ROCM_PATH_SEARCH" ]]; then
    ROCM_VER=$(rocm-smi --version 2>/dev/null | grep -oP '\d+\.\d+\.\d+' | head -1 || echo "installed")
    pass "ROCm found at: $ROCM_PATH_SEARCH (v$ROCM_VER)"
else
    warn "ROCm path not identified — tools still accessible via PATH"
fi

# ── 9. Optional: HIP compute test ────────────────────────────────────────────
hdr "HIP / Python Compute Test"

python3 - <<'PYEOF' 2>&1 | while read -r line; do
try:
    import subprocess, sys
    # Try to check if any HIP-capable library is available
    result = subprocess.run(['rocminfo'], capture_output=True, text=True)
    agents = [l for l in result.stdout.split('\n') if 'Device Type' in l and 'GPU' in l]
    if agents:
        print(f"PASS:rocminfo sees {len(agents)} GPU agent(s)")
    else:
        print("WARN:No GPU agents in rocminfo")
except Exception as e:
    print(f"WARN:{e}")

try:
    import ctypes
    hip = ctypes.cdll.LoadLibrary("libamdhip64.so")
    count = ctypes.c_int(0)
    hip.hipGetDeviceCount(ctypes.byref(count))
    print(f"PASS:HIP sees {count.value} device(s)")
except Exception as e:
    print(f"WARN:HIP library not directly accessible ({e})")

try:
    import torch
    if torch.cuda.is_available():
        print(f"PASS:PyTorch ROCm — {torch.cuda.get_device_name(0)}")
    else:
        v = torch.__version__
        if 'rocm' in v.lower():
            print(f"WARN:PyTorch ROCm build ({v}) but no device found — check HSA_OVERRIDE_GFX_VERSION")
        else:
            print(f"WARN:PyTorch ({v}) not built for ROCm — install rocm-compatible wheel")
except ImportError:
    print("WARN:PyTorch not installed (optional)")
PYEOF
    if [[ $line == PASS:* ]]; then pass "${line#PASS:}";
    elif [[ $line == FAIL:* ]]; then fail "${line#FAIL:}";
    else warn "${line#WARN:}"; fi
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "────────────────────────────────────────────────────────"
echo -e "${BOLD}Verification Summary${NC}"
echo -e "  ${GREEN}Passed: $PASS${NC}  ${YELLOW}Warnings: $WARN${NC}  ${RED}Failed: $FAIL${NC}"

if [[ $FAIL -eq 0 && $WARN -le 2 ]]; then
    echo -e "${GREEN}✓ ROCm is properly configured for LLM inference.${NC}"
elif [[ $FAIL -eq 0 ]]; then
    echo -e "${YELLOW}⚠ ROCm is functional but some optional features may improve performance.${NC}"
else
    echo -e "${RED}✗ $FAIL critical issue(s) need attention — see FAIL items above.${NC}"
fi
