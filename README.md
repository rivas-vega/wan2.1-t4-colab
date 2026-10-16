<div align="center">
  <img src="assets/hero_banner.svg" alt="Wan 2.1 Video Studio" width="100%" />
  <br/><br/>
  <a href="https://colab.research.google.com/github/rivas-vega/wan2.1-t4-colab/blob/main/wan2_1_t4_colab_free_inference.ipynb">
    <img src="https://img.shields.io/badge/Launch_Studio-Google_Colab_T4-0064e0?style=flat-square&logo=googlecolab&logoColor=white" alt="Open In Colab" />
  </a>
  <a href="https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers">
    <img src="https://img.shields.io/badge/Weights-Wan_2.1_T2V_1.3B-f1f5f9?style=flat-square&labelColor=0f172a&color=334155" alt="Model Weights" />
  </a>
  <img src="https://img.shields.io/badge/Target_Hardware-NVIDIA_T4_(15GB_VRAM_%2F_12.7GB_RAM)-f1f5f9?style=flat-square&labelColor=0f172a&color=334155" alt="Hardware" />
  <img src="https://img.shields.io/badge/Precision-FP16_DiT_%2B_NF4_UMT5_%2B_FP32_VAE-f1f5f9?style=flat-square&labelColor=0f172a&color=334155" alt="Precision" />
</div>

<br/>

## Overview

Modern open-weights video foundation models such as **Alibaba Wan 2.1** ship with a **22.7 GB FP32 text encoder (`UMT5-XXL`)**, a **1.3B 3D Diffusion Transformer (`DiT`)**, and a **3D Causal Spatiotemporal `VAE`**. Standard `diffusers` pipelines assume workstation-class hardware (32–64 GB of System RAM and 24+ GB of GPU VRAM) and immediately crash free **Google Colab T4 instances** (`12.7 GB System RAM`, `15.0 GB VRAM`).

This repository provides a **inference engine and minimalist web studio** engineered specifically for constrained cloud GPUs.

<table>
  <tr>
    <td width="38%" align="center" valign="middle">
      <br/>
      <img src="assets/demo_t4_vlog.gif" alt="Wan 2.1 T4 Generated Sample (Seed 342683004)" width="280" />
      <br/><br/>
      <sub>
        <b>Unedited Tesla T4 Output</b><br/>
        <code>480×832</code> · <code>65 frames (4.1s)</code> · <code>Seed 342683004</code><br/>
        <code>UniPC (flow_shift = 3.0, 45 steps, CFG = 5.0)</code>
      </sub>
      <br/>
    </td>
    <td width="62%" valign="top">
      <h3>Project</h3>
      <ul>
        <li><b>Zero-Cost Video Synthesis on Free Tier Hardware:</b> Enables researchers, creators, and developers to generate continuous <b>4.1-second (65-frame)</b> and <b>5.1-second (81-frame)</b> videos in native vertical (<code>9:16</code>) and widescreen (<code>16:9</code>) formats on a free Google Colab T4 without paid APIs or enterprise GPUs.</li>
        <li><b>True Live 3D Trajectory Preview:</b> Instead of generating a separate 2D preview image (which diverges from the 3D video due to full spatiotemporal attention), the studio decodes Frame 0 directly from the active solver's clean prediction ($\hat{x}_0^{(t)}$) at steps 8, 18, and 30 in <code>0.15s</code>—allowing immediate cancellation if framing or lighting is off.</li>
        <li><b>Reference Systems Architecture for Constrained Containers:</b> Demonstrates how to orchestrate >25 GB multimodal pipelines inside strict <code>12.7 GB Host RAM</code> limits via inverted GPU residency, C-level allocator trimming (<code>glibc mallopt</code>), and dynamic pre-VAE module eviction.</li>
        <li><b>Zero-VRAM Prompt Engineering:</b> Bridges the gap between short user inputs (in Spanish, English, or any language) and the dense 80-word optical captions required by Wan 2.1 without loading an auxiliary LLM into GPU memory.</li>
      </ul>
    </td>
  </tr>
</table>

---

## System Architecture

<div align="center">
  <img src="assets/pipeline_architecture.svg" alt="Wan 2.1 End-to-End Inference Pipeline" width="100%" />
</div>

---

## Engineering Challenges & Solutions

<div align="center">
  <img src="assets/memory_telemetry.svg" alt="Memory Telemetry Comparison on Google Colab T4" width="100%" />
</div>

<br/>

### 1. Inverted Memory Topology & Direct-to-GPU 4-Bit Streaming
In standard desktop environments, Host RAM is larger than GPU VRAM, making `enable_model_cpu_offload()` a viable strategy. **On Google Colab Free Tier, this relationship is inverted:** Host RAM is capped at **12.7 GB**, whereas the Tesla T4 provides **15.0 GB of VRAM**.
* **The Challenge:** Loading `UMT5-XXL` (22.7 GB across 6 FP32 safetensors shards) on the CPU or using Hugging Face's default Xet in-memory chunk reconstruction spikes System RAM to `13.9 GB`, triggering the Linux kernel OOM killer. Furthermore, re-running a notebook cell with `device_map="auto"` when VRAM is partially occupied causes `accelerate` to split 4-bit modules onto CPU/disk, raising a `ValueError`.
* **The Solution ([`pipeline_utils.py`](pipeline_utils.py)):**
  1. Disables Xet memory buffering (`HF_HUB_DISABLE_XET=1`) so shards stream directly from disk.
  2. Streams `UMT5EncoderModel` directly onto the GPU (`device_map={"": 0}`) using `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`, compressing the 22.7 GB encoder into **2.4 GB of VRAM** while keeping **System RAM at `3.0 / 12.7 GB`**.
  3. Tunes the `glibc` allocator at runtime via `ctypes` (`mallopt(M_MMAP_THRESHOLD, 1MB)` and `malloc_trim(0)`) so freed tensor buffers are returned immediately to the OS kernel.
  4. Implements **Idempotent Hot-VRAM Reuse**: re-running the loader detects active GPU weights in `sys.modules["__main__"]` and reuses the pipeline in `0.1s` without duplicating memory.

### 2. Pre-VAE Dynamic DiT Eviction for Long 65–81 Frame Decoding
* **The Challenge:** While the 1.3B DiT in Native FP16 (`2.6 GB`) easily processes 65–81 frames (`17–21` latent temporal slices) using memory-efficient Cutlass SDPA, the final **3D Causal VAE decoding step (`vae.decode`)** upsamples the 5D latent volume by $4\times 8\times 8$ in `float32`. Keeping the DiT resident in VRAM during 65-frame 3D VAE decoding pushes peak VRAM past 15.0 GB.
* **The Solution:** Inside `_step_callback`, at the exact completion of the final denoising step (`step_index == total_steps - 1`)—milliseconds before `vae.decode()` executes—the pipeline temporarily offloads `pipe.transformer` (`2.6 GB`) to the **9.7 GB of free System RAM** and flushes the CUDA cache. Combined with `vae.enable_tiling()` and `vae.enable_slicing()`, this opens **over 4.5 GB of extra VRAM headroom** exclusively for 3D VAE decoding, enabling full **4.1s (65 frames)** and **5.1s (81 frames)** continuous videos on a 15 GB T4.

### 3. Preserving High-Frequency Surface Textures at 480p (`flow_shift = 3.0`)
* **The Challenge:** Wan 2.1 uses Rectified Flow Matching with a shifted noise schedule:
  $$\sigma_{\text{shifted}}(t) = \frac{s \cdot \sigma(t)}{1 + (s - 1)\sigma(t)}$$
  Using the 720p shift factor ($s = 5.0$) at 480p resolution ($480 \times 832$) crowds 80% of the solver steps into the high-noise structural regime ($\sigma > 0.7$) and starves the low-noise regime ($\sigma \in [0.05, 0.35]$) where skin, fabric, hair, and environmental micro-textures are resolved. Additionally, mutating Classifier-Free Guidance (`guidance_scale`) mid-trajectory inside a second-order multistep solver (`UniPCMultistepScheduler`) corrupts finite-difference polynomial derivatives ($\hat{x}_0^{(t)} - \hat{x}_0^{(t-1)}$).
* **The Solution:** Configures `UniPCMultistepScheduler` with `prediction_type="flow_prediction"`, `use_flow_sigmas=True`, and the official 480p **`flow_shift = 3.0`** over **45 steps** with a strictly constant **`guidance_scale = 5.0`**, exporting clean `CRF 14` H.264 video without artificial post-processing noise.

### 4. Exact Live Frame Preview from the Active 3D Video Trajectory
* **The Challenge:** In a 3D Full-Attention architecture, generating a separate 1-frame preview (`num_frames = 1`) produces a completely different scene and subject than a 65-frame video (`num_frames = 65`) even with the same random seed, because temporal self-attention mixes all 17 latent frames from step 0.
* **The Solution ([`_decode_single_preview_frame`](pipeline_utils.py)):** At steps 8, 18, and 30 of the actual 3D video generation, the callback extracts the first temporal slice of the solver's current clean prediction `pipe.scheduler.model_outputs[-1][:, :, 0:1, :, :]`, unscales it with `latents_mean` / `latents_std`, and decodes that single frame via `pipe.vae` in `0.15s`. The user sees the exact video taking shape in real time and can abort early via the **Stop** button if desired.

---

## Generation Presets

| Preset | Resolution | Duration | Native Frames | Latent Shape ($T_L \times H_L \times W_L$) | Solver Configuration | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Reel 9:16 · 4s`** *(Default)* | `480 × 832` | **4.1 s** (`16 fps`) | `65` | `17 × 104 × 60` | `UniPC (flow_shift=3.0, 45 steps, CFG=5.0)` | `10.4 / 15.0 GB` |
| **`Reel 9:16 · 5s`** | `480 × 832` | **5.1 s** (`16 fps`) | `81` | `21 × 104 × 60` | `UniPC (flow_shift=3.0, 45 steps, CFG=5.0)` | `11.2 / 15.0 GB` |
| **`Cinema 16:9 · 4s`** | `832 × 480` | **4.1 s** (`16 fps`) | `65` | `17 × 60 × 104` | `UniPC (flow_shift=3.0, 45 steps, CFG=5.0)` | `10.4 / 15.0 GB` |

---

## Quick Start (Google Colab)

1. Open [`wan2_1_t4_colab_free_inference.ipynb`](https://colab.research.google.com/github/rivas-vega/wan2.1-t4-colab/blob/main/wan2_1_t4_colab_free_inference.ipynb) in Google Colab and select **Runtime → Change runtime type → T4 GPU**.
2. Run **Step 1 (Setup Environment)** to install dependencies and configure `HF_HUB_DISABLE_XET=1`.
3. *(Optional)* Enter a free Hugging Face read token in **Step 2** to enable `Qwen/Qwen2.5-72B-Instruct` cloud prompt conditioning and faster weight streaming.
4. Run **Step 3 (Preload Engine)** or launch **Step 4 (Wan 2.1 Studio)** directly—the Gradio web studio automatically initializes the pipeline on the GPU if not already loaded.

---

## Repository Structure

```text
├── assets/
│   ├── hero_banner.svg                      # Header graphic
│   ├── pipeline_architecture.svg            # 3-stage system architecture diagram
│   ├── memory_telemetry.svg                 # Colab RAM/VRAM telemetry comparison
│   ├── demo_t4_vlog.gif                     # Unedited 65-frame (4.1s) T4 output preview
│   └── demo_t4_vlog.mp4                     # Full-quality H.264 source video (Seed 342683004)
├── wan2_1_t4_colab_free_inference.ipynb     # 1-click Google Colab notebook
├── app.py                                   # Minimalist Gradio web studio with live x0 preview & Stop control
├── pipeline_utils.py                        # 4-bit NF4 GPU loader, Pre-VAE DiT eviction & UniPC(flow_shift=3.0)
├── requirements.txt                         # Python package dependencies
└── README.md                                # Technical documentation
```

---

## Author

**Pedro** ([@rivas-vega](https://github.com/rivas-vega))  
M2 MVA, École Normale Supérieure Paris-Saclay · B.S. in Mathematics, Universidad de Sevilla
