# Local LLM Optimization Guide
## AMD Ryzen AI MAX+ 395 + Radeon 8060S (gfx1151) + 128GB Unified Memory

---

## Your Hardware at a Glance

| Component | Spec | LLM Relevance |
|-----------|------|---------------|
| CPU | Ryzen AI MAX+ 395 (16c/32t, up to 5.1 GHz) | Fast prompt tokenization, CPU fallback |
| GPU | Radeon 8060S (gfx1151 / RDNA 3.5) | Primary compute for inference |
| Memory | 128GB LPDDR5X **unified** | GPU and CPU share this pool — this is your key advantage |
| Storage | 3.7TB NVMe | Fast model loading |
| ROCm | Installed + working | Full GPU acceleration in Ollama |

**Why unified memory matters:** Unlike discrete GPUs where VRAM is a hard ceiling (e.g. 24GB on an RTX 4090), your GPU can reach into the full 128GB pool. You can run models that would be impossible on much more expensive discrete setups — a 70B model fits comfortably.

---

## ROCm / Driver Configuration

### Enable XNACK for Best Unified Memory Performance

XNACK allows the GPU to recover from page faults in unified memory rather than crashing. This is important for APUs.

```bash
# Add to /etc/default/grub (GRUB_CMDLINE_LINUX):
amdgpu.noretry=0

# Or set at runtime (add to /etc/environment or ~/.profile):
HSA_XNACK=1
```

Then rebuild grub and reboot if you modified the kernel line.

### Verify GPU is Accessible

```bash
# Your user should be in the render and video groups:
sudo usermod -aG render,video $USER  # then re-login

# Confirm:
groups | grep -E "render|video"
```

### Override GFX Version (if a tool doesn't recognize gfx1151)

Some older ROCm tools may not recognize gfx1151. If you hit "unsupported GPU" errors:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.5.1
```

Add to `~/.profile` for permanence.

---

## Ollama Configuration

### Set Compute Mode to GPU

Ollama auto-detects ROCm. Confirm it's using GPU:

```bash
ollama ps  # should show "100% GPU" next to running model
```

### Tune Memory Allocation

```bash
# /etc/systemd/system/ollama.service.d/override.conf
[Service]
Environment="OLLAMA_MAX_LOADED_MODELS=2"      # keep 2 models hot
Environment="OLLAMA_NUM_PARALLEL=4"            # parallel request slots
Environment="OLLAMA_KEEP_ALIVE=30m"            # keep model in memory 30 min
Environment="OLLAMA_FLASH_ATTENTION=1"         # faster attention (ROCm supported)
```

Apply with: `sudo systemctl daemon-reload && sudo systemctl restart ollama`

### Context Window Tuning

Your 128GB pool can handle very large contexts. Set per-request:

```bash
ollama run qwen3:32b --context-size 32768
```

Or globally in Modelfile:
```
PARAMETER num_ctx 32768
```

Rough RAM cost: ~512MB per 8k tokens for a 32B model.

---

## Model Selection Strategy

### What Fits Comfortably (leave ~20GB headroom for OS + overhead)

| Model Size | RAM Used (Q4) | Recommendation |
|------------|---------------|----------------|
| 7-8B       | ~5 GB         | Lightning fast, good for agentic/code tasks |
| 14-15B     | ~9 GB         | Great quality/speed balance |
| 27-32B     | ~18-21 GB     | Excellent quality, still fast |
| 70B        | ~42 GB        | Fits well, excellent for complex reasoning |
| 120-130B   | ~75-80 GB     | Fits with room to spare — your superpower |

### Installed Models — Recommendations

| Model | Use Case | Expected Speed |
|-------|----------|----------------|
| `llama3.1:8b` | Fast chat, scripting | ~40-80 tok/s |
| `mistral-small` (14B equiv) | General purpose, fast | ~30-60 tok/s |
| `gemma3:27b` | High quality chat/analysis | ~20-40 tok/s |
| `qwen3:32b` | Reasoning, multilingual | ~18-35 tok/s |
| `qwen3.5:35b` | Code + reasoning | ~15-30 tok/s |
| `qwen3-coder` (18B) | Code generation | ~25-45 tok/s |
| `deepseek-r1:8b` | Fast reasoning | ~40-70 tok/s |
| `deepseek-r1:70b` | Deep reasoning | ~8-15 tok/s |

### Quantization Advice

- **Q4_K_M** — best quality/size trade-off, use by default
- **Q5_K_M** — slightly better quality, ~25% more RAM (worth it for 7-8B)
- **Q8_0** — near lossless, use when RAM allows and quality is critical
- **IQ2/IQ3** — aggressive compression, avoid for tasks requiring reasoning
- **F16** — only if you're doing fine-tuning or need exact weights

For your 128GB setup, prefer **Q5_K_M or Q8_0** for models up to 32B. For 70B, use **Q4_K_M**.

---

## Performance Tuning

### Ollama GPU Layers

Ollama automatically offloads as many layers as possible to GPU. To force full offload:

```bash
# In a Modelfile:
PARAMETER num_gpu 999  # offload all layers
```

### Flash Attention

ROCm supports Flash Attention 2, which reduces memory bandwidth for long contexts:

```bash
OLLAMA_FLASH_ATTENTION=1 ollama serve
```

### Batch Size for Throughput

If running multiple users or agentic workloads:

```bash
Environment="OLLAMA_NUM_PARALLEL=4"   # 4 concurrent inference slots
```

Each slot uses ~same VRAM as a single run, but amortizes model loading cost.

### Disable CPU Fallback (force GPU or fail)

```bash
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="ROCR_VISIBLE_DEVICES=0"
```

---

## Workflows That Shine on Your Hardware

### 1. Long Context Document Processing

Your 128GB pool lets you load massive contexts. Example:

```bash
cat large_document.txt | ollama run qwen3:32b "Summarize and extract key findings:"
```

### 2. Running Multiple Models Simultaneously

With 128GB, you can keep a fast model (8B) and a quality model (32B) both hot:

```bash
OLLAMA_MAX_LOADED_MODELS=2 ollama serve
```

Use the 8B for routing/classification, 32B for final generation.

### 3. Retrieval-Augmented Generation (RAG)

`nomic-embed-text` is already installed — fast embeddings on GPU. Pair with:
- [Ollama + Open WebUI](https://openwebui.com/) for a full local ChatGPT replacement
- LangChain / LlamaIndex for programmatic RAG pipelines

### 4. Code Generation Pipelines

`qwen3-coder` variants are installed. For agentic coding:
```bash
ollama run qwen3-coder "Review this code and suggest improvements:" < myfile.py
```

### 5. Local Reasoning / o1-style

`deepseek-r1:70b` provides o1-class reasoning fully locally — rare capability.

---

## Monitoring & Diagnostics

```bash
# Real-time GPU monitoring
watch -n 1 rocm-smi

# Memory pressure check
rocm-smi --showmeminfo vram

# Ollama metrics
curl http://localhost:11434/api/ps | python3 -m json.tool

# Token throughput from last run (in Ollama logs)
journalctl -u ollama -n 100 | grep "tokens/s"
```

---

## Known Issues & Workarounds

### PyTorch Not Recognizing GPU

PyTorch installed is CUDA-built (`cu128`), not ROCm-built. This doesn't affect Ollama (which uses its own ROCm runtime), but affects Python ML workflows.

To fix:
```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
# Or check: https://pytorch.org/get-started/locally/ → ROCm
```

Then verify:
```python
import torch
print(torch.cuda.is_available())  # should be True with ROCm build
print(torch.cuda.get_device_name(0))
```

### gfx1151 Compatibility

gfx1151 (Strix Halo) is relatively new. If you encounter "GPU not supported" in any tool:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.5.1
```

### VRAM Reported as 512MB by rocm-smi

This is expected for APUs. ROCm-SMI reports only the BIOS-allocated VRAM slice (~512MB). Ollama and ROCm runtime use HSA unified memory which can access the full 128GB. The "VRAM%" reading in rocm-smi is not meaningful for model capacity.

---

## Quick Reference Commands

```bash
# Run benchmark suite
./benchmark.sh                        # small + medium models
./benchmark.sh --quick                # just short prompts
./benchmark.sh --all                  # all installed models
./benchmark.sh --model qwen3:32b      # specific model

# Verify ROCm
./rocm_verify.sh

# Compare results
python3 bench_compare.py              # latest result
python3 bench_compare.py results/*.json  # all results

# Real-time GPU monitor
watch -n 0.5 rocm-smi --showuse
```
