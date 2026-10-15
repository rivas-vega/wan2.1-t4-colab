# Wan 2.1 Video Generation on Free Google Colab (Tesla T4 16GB)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rivas-vega/wan2.1-t4-colab/blob/main/wan2_1_t4_colab_free_inference.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Diffusers](https://img.shields.io/badge/HuggingFace-Diffusers-orange.svg)](https://github.com/huggingface/diffusers)
[![Model: Wan 2.1](https://img.shields.io/badge/Model-Wan2.1--1.3B-blue.svg)](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers)
[![GPU: Tesla T4](https://img.shields.io/badge/GPU-Tesla%20T4%20(15GB)-green.svg)]()

A hardware-optimized pipeline and interactive **Meta-style Gradio UI** to generate high-fidelity videos **with synchronized audio** using Alibaba's **Wan 2.1 (T2V-1.3B)** and **AudioLDM** on a **100% free Google Colab instance** (NVIDIA Tesla T4 GPU, 15 GB VRAM) in **~2.5 to 4.5 minutes**.

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

### 2. The Quantization Trap: Why 4-bit NF4 is Slower Than Native FP16 on Tesla T4

| Dimension | Native FP16 (2.0 bytes/weight) | 4-bit NF4 bitsandbytes (0.5 bytes/weight) |
| :--- | :--- | :--- |
| **DiT 1.3B Weight Size** | **2.6 GB** | **~750 MB** |
| **Hardware Path on T4** | Native Turing FP16 Tensor Cores (65 TFLOPS) | Dynamic software dequantization in CUDA registers |
| **Compute Overhead** | Zero dequantization latency | **~1.8x latency multiplier per linear layer** |
| **Peak VRAM with CPU Offload** | **~7.8 GB** (well within 15 GB) | ~4.5 GB |
| **Speed Recommendation on T4** | **⚡ FASTEST (~1.8x faster per step)** | Slower (only needed if VRAM < 8 GB) |

*The Insight:* Because Wan 2.1-1.3B's DiT is compact (**only 2.6 GB in FP16**), and because **Model CPU Offloading** moves the 9.6 GB UMT5 text encoder off GPU before the diffusion loop begins, **FP16 easily fits within the T4's 15 GB VRAM with 7+ GB of headroom**. Running 4-bit NF4 on Turing merely adds software dequantization penalties on every matrix multiplication.

---

### 3. Solver Optimization: UniPC vs Euler Flow Matching

Wan 2.1 is trained on continuous Rectified Flow. While standard `FlowMatchEulerDiscreteScheduler` requires 25–30 steps for convergence:
- **`UniPCMultistepScheduler` (with `flow_shift=3.0` for 480p/360p):** A higher-order predictor-corrector solver that converges in **14–16 steps** with identical structural fidelity.
- **Pass Multiplier:** With Classifier-Free Guidance (`guidance_scale=5.0`), each step requires 2 forward passes. Cutting steps from 25 down to 16 drops total forward passes from **50 down to 32** (a direct **36% time reduction**).

---

### 4. Memory Offloading Architecture Timeline

```
Timeline:
[Phase 1: Text Conditioning]
  ├── Host RAM:  DiT (2.6 GB) + VAE (1.5 GB) cached in CPU memory
  └── GPU VRAM:  UMT5-XXL (9.6 GB) loads -> encodes prompt -> OFFLOADS to Host RAM (0 GB VRAM)

[Phase 2: Spatiotemporal Denoising Loop (16 steps)]
  ├── Host RAM:  UMT5 (9.6 GB) + VAE (1.5 GB)
  └── GPU VRAM:  DiT 1.3B (2.6 GB) stays resident + SDPA attention activations (~3.5-5.0 GB)
                 Peak VRAM: ~7.6 GB (Safe margin: 7.4 GB remaining on T4)

[Phase 3: 3D Latent Decoding]
  ├── Host RAM:  UMT5 (9.6 GB) + DiT (2.6 GB)
  └── GPU VRAM:  3D Causal VAE (1.5 GB) with spatial/temporal tiling (Peak VRAM: < 4 GB)
```

---

## ⚡ Quick Start: Google Colab (1-Click)

Click the badge below to run directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rivas-vega/wan2.1-t4-colab/blob/main/wan2_1_t4_colab_free_inference.ipynb)

1. Verify GPU: **Runtime $\to$ Change runtime type $\to$ T4 GPU**.
2. Run **Cell 1 & 2** (installs diffusers with `hf-transfer` for fast downloads).
3. Run **Cell 3** (loads hardware-optimized FP16 pipeline with UniPC solver).
4. Run **Cell 4** for a quick 3-minute test, or **Cell 5** to launch the **Gradio Web Interface** with a shareable public URL (`share=True`).

---

## 🎛️ Minimal Meta-Style Web Application
 
Designed like modern frontier video tools (Meta Movie Gen / Llama): minimal prompt bar, inspiration chips, and simple length pills (`2s`, `3s`, `5s`). All hyperparameters (spatial resolution, latent tiling, UniPC steps, negative prompts, guidance scale) are pre-calibrated to optimal hardware settings so the user just types a prompt and hits **Create**.

- **🔊 Automatic Sound Track:** When enabled, synthesizes an acoustic track matching the video prompt via **AudioLDM** and seamlessly muxes it with ffmpeg (`-c:v copy -c:a aac`).
- **⏱️ 2s Clip (~4 min):** `624x352`, 33 frames (16 fps), 16 UniPC steps.
- **⏱️ 3s Clip (~4 min):** `512x320`, 49 frames (16 fps), 15 UniPC steps.
- **⏱️ 5s Full Clip (~9 min):** `512x320`, 81 frames (16 fps), 15 UniPC steps.

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
