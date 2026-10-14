# Wan 2.1 Video Generation on Free Google Colab (Tesla T4 16GB)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rivas-vega/wan2.1-t4-colab/blob/main/wan2_1_t4_colab_free_inference.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Diffusers](https://img.shields.io/badge/HuggingFace-Diffusers-orange.svg)](https://github.com/huggingface/diffusers)
[![Model: Wan 2.1](https://img.shields.io/badge/Model-Wan2.1--1.3B-blue.svg)](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers)
[![GPU: Tesla T4](https://img.shields.io/badge/GPU-Tesla%20T4%20(15GB)-green.svg)]()

A lightweight, reproducible pipeline and interactive **Gradio Web UI** to run Alibaba's state-of-the-art **Wan 2.1 (T2V-1.3B)** video generation model on a **100% free Google Colab instance** equipped with a standard **NVIDIA Tesla T4 GPU (15 GB VRAM)**.

---

## 🔬 The Memory Bottleneck & Technical Solution

### The Challenge: Out-Of-Memory (OOM) on 15 GB GPUs
Standard unoptimized execution of Wan 2.1-1.3B exceeds 16 GB of VRAM:
- **Text Encoder (UMT5-XXL 4.8B):** Consumes ~9.6–11.0 GB in FP16.
- **Diffusion Transformer (DiT 1.3B):** Consumes ~2.6 GB in FP16.
- **3D Causal VAE Decoder:** Consumes ~1.5 GB.
- **3D Spatio-Temporal Attention Activations:** Easily adds ~3–5 GB during 33–81 frame generation.
- **Total Unoptimized Footprint:** **> 16 GB $\to$ CUDA Out of Memory (OOM) crash on T4.**

### The Engineering Solution
By combining three complementary optimizations, this pipeline reduces peak VRAM to **~7.8 GB**, running comfortably within the free 15 GB Colab quota:

1. **Model CPU Offloading (`pipe.enable_model_cpu_offload()`):**
   The heavy 4.8B UMT5-XXL text encoder is moved to GPU only to encode the prompt text, then immediately offloaded to host system RAM before the 1.3B DiT begins denoising.
2. **3D VAE Tiling & Slicing (`pipe.vae.enable_tiling()` / `enable_slicing()`):**
   Prevents memory spikes when decoding the 3D latent volume back into RGB video frames by processing them in spatial and temporal chunks.
3. **Native FP16 Precision with PyTorch SDPA:**
   Leverages Turing Tensor Cores and Scaled Dot-Product Attention for optimal memory efficiency.

```
       [Prompt Input]
             │
      ┌──────▼──────┐  Loads into VRAM (~11 GB)
      │  UMT5-XXL   │  Generates text conditioning
      └──────┬──────┘  Immediately OFFLOADS to Host RAM
             │
      ┌──────▼──────┐  Loads into VRAM (~2.6 GB)
      │  1.3B DiT   │  Runs 25-30 diffusion denoising steps
      └──────┬──────┘  Peak VRAM during attention: ~7.8 GB
             │
      ┌──────▼──────┐  Tiled decoding in temporal slices
      │   3D VAE    │  Memory bounded: < 1.5 GB
      └──────┬──────┘
             │
      [Video Output (MP4)]
```

---

## ⚡ Quick Start: 1-Click Google Colab

Click the badge below to open the notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rivas-vega/wan2.1-t4-colab/blob/main/wan2_1_t4_colab_free_inference.ipynb)

1. Make sure GPU acceleration is active: **Runtime $\to$ Change runtime type $\to$ T4 GPU**.
2. Run **Cell 1 & 2** to install dependencies.
3. Run **Cell 3** to load the memory-optimized pipeline.
4. Run **Cell 5** to launch the **Gradio Web Interface** with a public link (`share=True`), allowing anyone to generate videos directly from their browser or phone.

---

## 💻 Local / Self-Hosted Execution

If you have a local machine or server with a 12GB–16GB NVIDIA GPU:

```bash
git clone https://github.com/rivas-vega/wan2.1-t4-colab.git
cd wan2.1-t4-colab

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch Gradio interface
python app.py
```

The terminal will print a local URL (`http://127.0.0.1:7860`) and a public shareable URL (`https://xxxx.gradio.live`).

---

## 🎛️ Gradio Web Interface Features

- **Prompt & Negative Prompt:** Full control over generation aesthetics with built-in quality filters.
- **Resolution Presets:**
  - `832x480` (16:9 Landscape - cinematic widescreen).
  - `480x832` (9:16 Portrait - social / reels format).
  - `624x624` (1:1 Square).
- **Frame Control:** Adjustable frame depth (17 frames for ~1s quick preview, 33 frames for ~2s standard video, 49 frames for ~3s).
- **Sampling Steps & Guidance:** Sliders for denoising steps (default 25) and CFG scale (default 5.0).
- **Reproducibility:** Seed control with a one-click randomizer button.
- **Video Player:** Integrated web player with looping and MP4 download button.

---

## 📁 Repository Structure

```text
├── wan2_1_t4_colab_free_inference.ipynb   # Complete, runnable Google Colab notebook
├── app.py                                # Standalone Gradio Web UI application
├── pipeline_utils.py                     # Memory offload & VAE tiling loader utilities
├── requirements.txt                      # Python dependencies
└── README.md                             # Documentation & architecture breakdown
```

---

## 👤 Author

**Pedro** ([@rivas-vega](https://github.com/rivas-vega))  
*M2 MVA, École Normale Supérieure Paris-Saclay*  
*B.S. in Mathematics, Universidad de Sevilla*
