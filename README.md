# Wan 2.1 Video Generation on Free Google Colab (Tesla T4 16GB)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rivas-vega/wan2.1-t4-colab/blob/main/wan2_1_t4_colab_free_inference.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Diffusers](https://img.shields.io/badge/HuggingFace-Diffusers-orange.svg)](https://github.com/huggingface/diffusers)
[![Model: Wan 2.1](https://img.shields.io/badge/Model-Wan2.1--1.3B-blue.svg)](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers)
[![GPU: Tesla T4](https://img.shields.io/badge/GPU-Tesla%20T4%20(15GB)-green.svg)]()

A hardware-optimized pipeline and interactive **Meta-style Gradio UI** to generate high-fidelity videos using Alibaba's **Wan 2.1 (T2V-1.3B)** on a **100% free Google Colab instance** (NVIDIA Tesla T4 GPU, 15 GB VRAM) with **Native FP16 DiT**, **4-bit UMT5 compression**, and **UniPC higher-order scheduling**.

---

## 🔬 Mathematical & Hardware Performance Analysis

### 1. The Core Bottleneck: Why Does Unoptimized Wan 2.1 Take >20 Minutes on a T4?

Executing Wan 2.1 on an NVIDIA Tesla T4 involves two primary hardware constraints: **memory capacity (15 GB)** and **arithmetic throughput (Turing architecture, compute capability 7.5)**.

#### A. Spatiotemporal Latent Token Scaling $\mathcal{O}(S^2)$
Wan 2.1's 3D causal VAE downsamples inputs spatially by $8\times$ and temporally by $4\times$. For a video of dimensions $W \times H$ with $F$ frames, the resulting 3D latent grid shape $[T_L, H_L, W_L]$ produces a sequence length $S$:

$$T_L = \left\lfloor\frac{F - 1}{4}\right\rfloor + 1, \quad H_L = \frac{H}{8}, \quad W_L = \frac{W}{8}$$

$$S = T_L \times H_L \times W_L$$

The computational complexity of 3D multi-head self-attention scales quadratically with sequence length $S$:

$$\text{FLOPs}_{\text{attention}} \propto 30 \times 4 \times S^2 \times d \quad (d = 1536)$$

| Resolution ($W \times H$) | Frames ($F$) | Latents $[T_L \times H_L \times W_L]$ | Sequence Length $S$ | Relative Attention Complexity $S^2$ | Measured T4 Time (FP16) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$832 \times 480$** (16:9 High) | 33 (~2s) | $[9 \times 60 \times 104]$ | **56,160 tokens** | $3.15 \times 10^9$ (**1.00x**) | ~12–16 min |
| **$832 \times 480$** (16:9 Std) | 25 (~1.5s) | $[7 \times 60 \times 104]$ | **43,680 tokens** | $1.91 \times 10^9$ (**0.61x**) | ~7–9 min |
| **$624 \times 352$** (Balanced) | 25 (~1.5s) | $[7 \times 44 \times 78]$ | **24,024 tokens** | $0.58 \times 10^9$ (**0.18x**) | **~2.8–3.5 min** |
| **$512 \times 320$** (Fast Draft) | 17 (~1s) | $[5 \times 40 \times 64]$ | **12,800 tokens** | $0.16 \times 10^9$ (**0.05x**) | **~1.2–1.5 min** |

> Reducing from $832 \times 480$ (33 frames) to $624 \times 352$ (25 frames) cuts quadratic attention operations by **$5.4\times$**, slashing generation time from ~16 minutes down to under 3 minutes.

---

### 2. The Memory Architecture: Why GPU-Resident DiT + 4-bit UMT5 Eliminates Host RAM OOM

| Component | Precision | Size | Device | Hardware Role |
| :--- | :--- | :--- | :--- | :--- |
| **UMT5-XXL Text Encoder** | 4-bit NF4 (`bitsandbytes`) | **2.4 GB** | `cuda:0` | Shards streamed directly to GPU, 0 GB Host RAM footprint |
| **Wan 2.1 DiT 1.3B** | Native FP16 (torch.float16) | **2.6 GB** | `cuda:0` | Native Turing FP16 Tensor Cores (65 TFLOPS, no dequant latency) |
| **3D Causal VAE** | FP32 (spatial/temporal tiling) | **~480 MB** | `cuda:0` | Overlapping latent chunk decoding, peak VRAM < 3 GB |
| **Total Static Footprint** | Combined | **5.5 GB** | **GPU VRAM** | **Well within 15.0 GB T4 limit (9.5 GB free headroom)** |
| **Host System RAM** | Linux + Python runtime | **~2.5 GB** | **Host RAM** | **Well within 12.7 GB Colab limit (10 GB free margin)** |

*The Architectural Insight:* In Google Colab, **GPU VRAM (15.0 GB) is larger than Host RAM (12.7 GB)**! Relying on standard CPU offloading is backwards because it forces 14+ GB of models into a constrained 12.7 GB Host RAM, triggering swap and kernel crashes. By streaming UMT5 directly into 4-bit NF4 on the GPU and keeping the compact 1.3B DiT in native FP16, **the entire model resides on the GPU in only 5.5 GB**, preserving Host RAM at ~2.5 GB and eliminating PCIe weight-swapping overhead.

---

### 3. Solver Optimization: UniPC vs Euler Flow Matching

Wan 2.1 is trained on continuous Rectified Flow. While standard `FlowMatchEulerDiscreteScheduler` requires 25–30 steps for convergence:
- **`UniPCMultistepScheduler` (with `flow_shift=3.0` for 480p/360p):** A higher-order predictor-corrector solver that converges in **14–22 steps** with identical structural fidelity.
- **Pass Multiplier:** With Classifier-Free Guidance (`guidance_scale=5.0`), each step requires 2 forward passes. Cutting steps from 30 down to 20 drops total forward passes from **60 down to 40** (a direct **33% latency reduction**).

---

### 4. GPU VRAM Timeline (Zero Host RAM Spikes)

```text
Timeline:
[Phase 1: Text Conditioning]
  ├── Host RAM:  ~2.5 GB (Python kernel + Gradio server)
  └── GPU VRAM:  4-bit UMT5 (2.4 GB) encodes prompt -> outputs prompt_embeds (Peak: ~2.8 GB)

[Phase 2: Spatiotemporal Denoising Loop (20-22 steps)]
  ├── Host RAM:  ~2.5 GB (Static, 0 GB memory traffic)
  └── GPU VRAM:  DiT 1.3B FP16 (2.6 GB) + UMT5 (2.4 GB) + SDPA attention activations (~3.5-4.0 GB)
                 Peak VRAM: ~9.0 GB (Safe margin: 6.0 GB remaining on T4)

[Phase 3: 3D Latent Decoding]
  ├── Host RAM:  ~2.5 GB (Static)
  └── GPU VRAM:  3D Causal VAE (0.5 GB) with spatial/temporal tiling (Peak VRAM: < 6.5 GB)
```

---

## ⚡ Quick Start: Google Colab (1-Click)

Click the badge below to run directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rivas-vega/wan2.1-t4-colab/blob/main/wan2_1_t4_colab_free_inference.ipynb)

1. Verify GPU: **Runtime $\to$ Change runtime type $\to$ T4 GPU**.
2. Run **Cell 1 & 2** (installs dependencies with direct-to-disk streaming).
3. Run **Cell 3** (loads hardware-optimized Wan 2.1 in Native FP16 DiT + 4-bit UMT5).
4. Run **Cell 4** to launch the interactive **Meta-style Gradio UI** (`share=True`), or **Cell 5** to generate directly inside the notebook.

---

## 🎛️ Minimal Meta-Style Web Application
 
Designed like modern frontier video tools (Meta Movie Gen / Llama): minimal prompt bar, inspiration chips, and simple quality/length pills. All hyperparameters (spatial resolution, latent tiling, UniPC steps, negative prompts, guidance scale) are pre-calibrated to optimal hardware settings so the user just types a prompt and hits **Create**.

- **🎬 Cinematic 480p (HQ) [Recommended]:** `832x480` (16:9 native 480p), 25 frames (16 fps), 22 UniPC steps (~6.5 min on T4). Maximum visual fidelity, sharp textures, realistic lighting.
- **⚡ Widescreen Fast (1.5s):** `624x352` (16:9 widescreen), 25 frames (16 fps), 20 UniPC steps (~3.5 min on T4). Fast preview with clean aspect ratio.
- **⏱️ Extended Motion (3.0s):** `624x352` (16:9 widescreen), 49 frames (16 fps), 20 UniPC steps (~6.0 min on T4). Sustained temporal motion.

### Running Locally / On-Premise

```bash
git clone https://github.com/rivas-vega/wan2.1-t4-colab.git
cd wan2.1-t4-colab

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Launch Gradio interface (prints local http://127.0.0.1:7860 and public gradio.live link)
python3 app.py
```

---

## 📁 Repository Structure

```text
├── wan2_1_t4_colab_free_inference.ipynb   # 1-Click interactive Colab notebook with fast defaults
├── app.py                                # Gradio Web App with dynamic token & runtime estimator
├── pipeline_utils.py                     # Turing sm_75 CUDA optimizations, UniPC loader, & metrics
├── requirements.txt                      # Dependencies including hf-transfer
├── .gitignore                            # Cache and media ignore patterns
└── README.md                             # Mathematical derivation and documentation
```

---

## 👤 Author

**Pedro** ([@rivas-vega](https://github.com/rivas-vega))  
*M2 MVA, École Normale Supérieure Paris-Saclay*  
*B.S. in Mathematics, Universidad de Sevilla*
