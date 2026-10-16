# Wan 2.1 Video Generation on Free Google Colab (Tesla T4 16GB)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rivas-vega/wan2.1-t4-colab/blob/main/wan2_1_t4_colab_free_inference.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Diffusers](https://img.shields.io/badge/HuggingFace-Diffusers-orange.svg)](https://github.com/huggingface/diffusers)
[![Model: Wan 2.1](https://img.shields.io/badge/Model-Wan2.1--1.3B-blue.svg)](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers)
[![GPU: Tesla T4](https://img.shields.io/badge/GPU-Tesla%20T4%20(15GB)-green.svg)]()

A hardware-optimized pipeline and interactive **Meta-style Gradio UI** to generate high-fidelity videos using Alibaba's **Wan 2.1 (T2V-1.3B)** on a **100% free Google Colab instance** (NVIDIA Tesla T4 GPU, 15 GB VRAM) with **Native FP16 DiT**, **4-bit UMT5 compression**, **FlowMatchEuler (anti-morphing shift=3.0)**, and native **Reel 9:16 vertical format**.

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
| **$480 \times 832$** (Reel 9:16 HQ) | 25 (~1.5s) | $[7 \times 104 \times 60]$ | **43,680 tokens** | $1.91 \times 10^9$ (**0.61x**) | **~3.8–4.2 min** |
| **$832 \times 480$** (16:9 Cinema) | 25 (~1.5s) | $[7 \times 60 \times 104]$ | **43,680 tokens** | $1.91 \times 10^9$ (**0.61x**) | **~3.8–4.2 min** |
| **$480 \times 832$** (Reel 9:16 Ext) | 49 (~3.0s) | $[13 \times 104 \times 60]$ | **81,120 tokens** | $6.58 \times 10^9$ (**2.09x**) | **~7.0–7.5 min** |
| **$624 \times 352$** (Fast Draft) | 25 (~1.5s) | $[7 \times 44 \times 78]$ | **24,024 tokens** | $0.58 \times 10^9$ (**0.18x**) | **~2.8–3.2 min** |

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

### 3. Solver Optimization & Geometric Rigidity: Why FlowMatchEuler Eliminates "AI Jelly" Morphing

Wan 2.1 is trained on continuous Rectified Flow. In video diffusion models, rigid objects (cars, architecture, furniture, faces) are prone to rubbery deformations ("AI jelly" effect) when the trajectory solver overshoots flow gradients:
- **The Problem with High-Order Solvers (UniPC) on 1.3B:** `UniPCMultistepScheduler` uses polynomial predictor-corrector interpolation. In compact 1.3B models, subtle noise in velocity predictions causes polynomial curve fitting to oscillate across consecutive frames. This manifests as liquid-like morphing, melting surfaces, and shifting geometry.
- **The Solution: First-Order `FlowMatchEulerDiscreteScheduler(shift=3.0)`:** Euler updates follow the direct rectified flow velocity:
  $$x_{t-\Delta t} = x_t - \Delta t \cdot v_\theta(x_t, t, c)$$
  By taking strictly linear updates along the optimal transport vector field, Euler eliminates polynomial overshooting and enforces rigid geometric boundaries. Setting `shift=3.0` concentrates sampling density in high-frequency detail steps, restoring sharp texture resolution.
- **Guidance Scale ($\text{CFG}=5.5$):** Increasing guidance from 5.0 to 5.5 sharpens object boundaries and prevents texture smudging without color oversaturation.
- **28 Sampling Steps:** 28 Euler steps hit the Pareto sweet spot on T4 (28 steps $\times$ 2 passes = 56 forward passes in ~4.0 min), yielding crisp details and solid physics.

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
 
Designed like modern frontier video tools (Meta Movie Gen / Llama): minimal prompt bar, inspiration chips, centered vertical Reel stage, and simple preset pills. All hyperparameters (spatial resolution, latent tiling, 28 Euler steps, anti-morphing negative prompt, guidance scale 5.5) are pre-calibrated to optimal hardware settings so the user just types a prompt and hits **Create**.

- **📱 Reel 9:16 (HQ 480p) [Default / Recommended]:** `480x832` (9:16 vertical TikTok/Reel), 25 frames (16 fps), 28 Euler steps (~4.0 min on T4). Maximum geometric rigidity, sharp textures, solid reflections.
- **⏱️ Reel 9:16 (Extended 3s):** `480x832` (9:16 vertical), 49 frames (16 fps), 26 Euler steps (~7.0 min on T4). Extended temporal motion.
- **🎬 Cinema 16:9 (Landscape):** `832x480` (16:9 widescreen), 25 frames (16 fps), 28 Euler steps (~4.0 min on T4). High-fidelity cinematic landscape.

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
