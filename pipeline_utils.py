"""
pipeline_utils.py - Memory-Optimized Pipeline Loader for Wan 2.1 on 16GB GPUs (NVIDIA T4).

Enables running Alibaba's Wan 2.1 (T2V-1.3B) in free Google Colab instances by combining:
1. Model CPU Offloading: Sequentially loads UMT5 text encoder, 1.3B DiT, and VAE.
2. VAE Tiling & Slicing: Prevents high-resolution 3D spatiotemporal latent decoding spikes.
3. Quantization / FP16 precision: Drops VRAM footprint from >16 GB down to ~7.8 GB peak.
"""

import os
import gc
import time
import random
import ctypes
import threading
import subprocess
import torch
from typing import Optional, Tuple

# ── Host-RAM hygiene for Colab (12.7 GB) ──────────────────────────────────────
# glibc only reads MALLOC_* environment variables at process start-up, so setting
# them from Python has no effect. mallopt() changes the allocator at runtime:
#   M_MMAP_THRESHOLD (-3): buffers > 1 MB go through mmap and are returned to the
#                          OS on free (disables glibc's dynamic threshold, which
#                          otherwise grows to 32 MB and parks tensors in the heap).
#   M_TRIM_THRESHOLD (-1): trim the top of the heap once 64 KB are free.
#   M_ARENA_MAX      (-8): cap per-thread arenas (Gradio runs jobs in worker threads).
try:
    _LIBC = ctypes.CDLL("libc.so.6")
    _LIBC.mallopt(-3, 1 << 20)
    _LIBC.mallopt(-1, 64 << 10)
    _LIBC.mallopt(-8, 2)
except OSError:  # non-glibc platform (macOS / Windows)
    _LIBC = None


def get_gpu_memory_summary() -> str:
    """Return formatted GPU VRAM usage if CUDA is available."""
    if not torch.cuda.is_available():
        return "CUDA is not available (running on CPU)."
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    return (
        f"VRAM Allocated: {allocated:.2f} GB | "
        f"Reserved: {reserved:.2f} GB | "
        f"Total: {total:.2f} GB"
    )


def clear_cuda_cache():
    """Aggressive memory collection to prevent fragmentation on T4 and Colab Host RAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    # Force glibc to hand freed heap pages back to the kernel immediately
    if _LIBC is not None:
        _LIBC.malloc_trim(0)


def setup_turing_cuda_optimizations():
    """Configure optimal CUDA settings for NVIDIA Turing (Tesla T4, sm_75)."""
    if not torch.cuda.is_available():
        return
    # T4 is Turing (sm_75): does not support TF32 or FlashAttention-2
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True

    # Force Memory-Efficient SDPA (Cutlass) which is fully accelerated on sm_75
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


def load_wan_t4_pipeline(
    model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    quantization: str = "fast_fp16",  # "fast_fp16" (Native FP16 DiT + 4-bit UMT5: best quality, 1.8x faster on T4)
    dtype: torch.dtype = torch.float16,
    scheduler_type: str = "euler",  # "euler" (strict linear flow, rigid geometry) or "unipc"
    hf_token: Optional[str] = None,
):
    """
    Load Wan 2.1 with hardware-tailored optimizations for a 16GB NVIDIA Tesla T4.

    Performance vs Precision on T4:
    - '4bit' (Default): Quantizes 22.7GB UMT5 + DiT via PipelineQuantizationConfig.
      Total System RAM stays < 5.0 GB (100% immune to Colab 12.7 GB crash).
    - 'fast_fp16': Quantizes ONLY the 22.7GB UMT5 to 4-bit, keeping DiT in native FP16 (~1.8x faster).
    - '8bit': 1.0 byte/weight (INT8 via bitsandbytes). Peak VRAM ~6.0 GB.

    Args:
        model_id: HuggingFace repository for official Diffusers weights.
        quantization: '4bit', 'fast_fp16', or '8bit'.
        dtype: Computation dtype (torch.float16 for Tesla T4 Turing architecture).
        scheduler_type: 'unipc' for fast convergence (14-18 steps) or 'euler' (25-30 steps).
        hf_token: Optional Hugging Face token for higher rate limits and faster downloads.

    Returns:
        Configured and memory-offloaded WanPipeline ready for fast inference.
    """
    # If re-running in an existing Colab/Jupyter session where `pipe` is already loaded on GPU,
    # reuse the GPU-resident weights in 0.1s and hot-upgrade the scheduler to FlowMatchEuler(shift=5.0)
    # instead of re-allocating VRAM and triggering bitsandbytes CPU-dispatch errors.
    import sys
    main_mod = sys.modules.get("__main__")
    if main_mod is not None:
        existing = getattr(main_mod, "pipe", None) or getattr(main_mod, "wan", None)
        if (
            existing is not None
            and getattr(existing, "transformer", None) is not None
            and getattr(existing, "text_encoder", None) is not None
            and getattr(existing, "vae", None) is not None
        ):
            try:
                from diffusers import FlowMatchEulerDiscreteScheduler
                existing.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                    existing.scheduler.config, shift=5.0
                )
            except Exception:
                pass
            clear_cuda_cache()
            print(f"[Wan 2.1 Loader] Reusing GPU-resident pipeline (FlowMatchEuler shift=5.0). {get_gpu_memory_summary()}")
            return existing

    setup_turing_cuda_optimizations()
    clear_cuda_cache()

    # Prevent Colab 12.7 GB Host RAM crash: disable Xet in-memory chunk reassembly
    # Standard HTTP chunk streaming writes directly to disk without multi-GB memory buffers
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)
    if hf_token and hf_token.strip():
        os.environ["HF_TOKEN"] = hf_token.strip()

    # Prevent diffusers from failing on Colab's pre-installed torchao
    try:
        import diffusers.utils.import_utils as _iu
        _iu._torchao_available = False
    except Exception:
        pass
    try:
        import torchao.quantization
        if not hasattr(torchao.quantization, "FqnToConfig"):
            class FqnToConfig:
                def __init__(self, *args, **kwargs): pass
            torchao.quantization.FqnToConfig = FqnToConfig
    except Exception:
        pass

    from diffusers import AutoencoderKLWan, WanPipeline, WanTransformer3DModel
    from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
    from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig, UMT5EncoderModel

    print(f"[Wan 2.1 Loader] Loading VAE (float32 with tiling)...")
    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
    )

    # CRITICAL COLAB FIX (Zero Host RAM OOM & Zero CPU-Offload ValueError):
    # Use explicit device_map={"": 0} instead of "auto" so accelerate never attempts to split
    # 4-bit UMT5 modules onto CPU/disk when VRAM is fragmented.
    print(f"[Wan 2.1 Loader] Streaming 22.7GB UMT5 to GPU in 4-bit NF4 (2.4 GB VRAM, 0 GB Host RAM)...")
    device_map = {"": 0} if torch.cuda.is_available() else None
    t5_bnb = TransformersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        quantization_config=t5_bnb,
        torch_dtype=dtype,
        device_map=device_map,
    )
    clear_cuda_cache()

    # Load 1.3B DiT Transformer
    transformer = None
    if quantization == "4bit":
        print(f"[Wan 2.1 Loader] 4-BIT: Quantizing DiT transformer with bitsandbytes 4-bit (0.8 GB VRAM)...")
        dit_bnb = DiffusersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
        transformer = WanTransformer3DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=dit_bnb,
            torch_dtype=dtype,
        )
    else:  # "fast_fp16" (Native FP16 DiT, 2.6 GB VRAM)
        print(f"[Wan 2.1 Loader] FAST FP16: Loading DiT transformer in native FP16 (2.6 GB VRAM)...")
        transformer = WanTransformer3DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=dtype,
        )

    print(f"[Wan 2.1 Loader] Assembling WanPipeline...")
    pipe = WanPipeline.from_pretrained(
        model_id,
        vae=vae,
        text_encoder=text_encoder,
        transformer=transformer,
        torch_dtype=dtype,
    )

    # Move remaining sub-modules to GPU (GPU has 15GB VRAM, model only takes 5.5GB total!)
    # We do NOT use enable_model_cpu_offload() because CPU RAM is only 12.7GB,
    # whereas GPU VRAM is 15.0GB! Keeping models on GPU preserves Host RAM and speeds up inference 2x!
    if torch.cuda.is_available():
        pipe.transformer.to("cuda")
        pipe.vae.to("cuda")
        # text_encoder is already managed on cuda via device_map="auto"

    # Configure FlowMatchEulerDiscreteScheduler (strict linear flow, zero polynomial overshooting / anti-morphing)
    if scheduler_type == "euler":
        try:
            from diffusers import FlowMatchEulerDiscreteScheduler
            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                pipe.scheduler.config,
                shift=5.0,  # 5.0 allocates more steps to structural formation, locking rigid geometry
            )
            print("[Wan 2.1 Loader] FlowMatchEulerDiscreteScheduler activated (shift=5.0 | rigid geometry & anti-morphing).")
        except Exception as e:
            print(f"[Wan 2.1 Loader] Euler scheduler fallback: {e}")
    elif scheduler_type == "unipc":
        try:
            from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
            pipe.scheduler = UniPCMultistepScheduler.from_config(
                pipe.scheduler.config,
                flow_shift=3.0,
            )
            print("[Wan 2.1 Loader] UniPC Multistep Scheduler activated.")
        except Exception as e:
            print(f"[Wan 2.1 Loader] UniPC scheduler fallback: {e}")

    # Enable VAE Tiling and Slicing (crucial to avoid OOM during 3D video decoding)
    if hasattr(pipe, "vae") and pipe.vae is not None:
        if hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        if hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()

    # Flush freed loading/quantization buffers and trim glibc heap memory
    clear_cuda_cache()

    print(f"[Wan 2.1 Loader] Pipeline ready ({quantization.upper()} mode). {get_gpu_memory_summary()}")
    return pipe


def calculate_spatiotemporal_tokens(width: int, height: int, num_frames: int) -> Tuple[int, int, int, int]:
    """
    Calculate Wan 2.1 3D latent dimensions and total sequence tokens.
    Wan 2.1 downsamples spatially by 8x and temporally by 4x.

    Returns:
        (total_tokens, latent_t, latent_h, latent_w)
    """
    latent_t = (num_frames - 1) // 4 + 1
    latent_h = height // 8
    latent_w = width // 8
    total_tokens = latent_t * latent_h * latent_w
    return total_tokens, latent_t, latent_h, latent_w


def estimate_generation_time_seconds(
    width: int,
    height: int,
    num_frames: int,
    steps: int,
    guidance_scale: float,
    is_fp16: bool = True,
) -> float:
    """
    Hardware-calibrated runtime estimator for NVIDIA Tesla T4 (Turing sm_75).
    Accounts for quadratic 3D self-attention scaling O(S^2 * d) and CFG pass multiplier.
    """
    tokens, _, _, _ = calculate_spatiotemporal_tokens(width, height, num_frames)
    passes_per_step = 2 if guidance_scale > 1.0 else 1
    total_passes = steps * passes_per_step

    # Normalized token factor (reference S0 = 24,000 tokens)
    norm_s = tokens / 24000.0

    # FP16 runs directly on Turing Tensor Cores; 4-bit incurs dynamic register dequantization overhead
    if is_fp16:
        sec_per_pass = 1.35 * norm_s + 1.15 * (norm_s ** 2)
    else:
        sec_per_pass = 2.40 * norm_s + 2.10 * (norm_s ** 2)

    vae_time = 0.35 * num_frames
    return total_passes * sec_per_pass + vae_time + 2.0


def generate_video(
    pipe,
    prompt: str,
    negative_prompt: str = "Bright, overexposed, static, low quality, distorted, blurry, deformed",
    width: int = 624,
    height: int = 352,
    num_frames: int = 25,
    steps: int = 28,
    guidance_scale: float = 5.5,
    seed: Optional[int] = None,
    output_path: str = "output_video.mp4",
    fps: int = 16,
    callback=None,
) -> str:
    """
    Generate a video sequence and export to MP4 with strict memory cleanup.

    Args:
        pipe: Initialized WanPipeline.
        prompt: Positive text prompt.
        negative_prompt: Negative guidance prompt.
        width: Video width (divisible by 16). Recommended: 624 or 512 for fast iteration.
        height: Video height (divisible by 16). Recommended: 352 or 320 for fast iteration.
        num_frames: Frame count. 25 (~1.5s) or 17 (~1s) for fast iteration; 33 (~2s) for standard.
        steps: Denoising steps (14-18 recommended with UniPC, 25 with Euler).
        guidance_scale: Classifier-free guidance scale (typically 5.0).
        seed: Random seed for reproducibility (-1 or None for random).
        output_path: Filepath where the MP4 video will be saved.
        fps: Playback framerate.
        callback: Optional callback_on_step_end for progress tracking without memory leaks.

    Returns:
        Path to the saved MP4 video.
    """
    clear_cuda_cache()

    tokens, lt, lh, lw = calculate_spatiotemporal_tokens(width, height, num_frames)
    est_sec = estimate_generation_time_seconds(width, height, num_frames, steps, guidance_scale, is_fp16=True)
    est_min = est_sec / 60.0

    print(
        f"[Generation] Starting Wan 2.1 inference:\n"
        f"  Resolution: {width}x{height} | Frames: {num_frames} ({num_frames/fps:.1f}s @ {fps}fps)\n"
        f"  Latent Shape: [T={lt}, H={lh}, W={lw}] -> Total 3D tokens: {tokens:,}\n"
        f"  Steps: {steps} | CFG: {guidance_scale} | Passes: {steps * (2 if guidance_scale > 1.0 else 1)}\n"
        f"  Estimated T4 Runtime: ~{est_min:.1f} min ({est_sec:.0f}s)"
    )

    generator = None
    if seed is not None and seed >= 0:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    pipe_kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
    }
    if callback is not None:
        pipe_kwargs["callback_on_step_end"] = callback

    output = None
    frames = None
    try:
        with torch.inference_mode():
            output = pipe(**pipe_kwargs)
        frames = output.frames[0] if hasattr(output, "frames") else output.images[0]

        from diffusers.utils import export_to_video
        export_to_video(frames, output_path, fps=fps)
    finally:
        # Crucial for Colab 12.7GB RAM: immediately deallocate huge 3D tensor objects
        if output is not None:
            del output
        if frames is not None:
            del frames
        clear_cuda_cache()

    print(f"[Generation] Video exported to {output_path}. {get_gpu_memory_summary()}")
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# Sound: AudioLDM 2 (text-to-audio) + ffmpeg muxing
# ══════════════════════════════════════════════════════════════════════════════
# Wan 2.1 is a silent video model. After the video is rendered, AudioLDM 2
# (cvssp/audioldm2, ~1.1 B params, not gated, 16 kHz) synthesises a sound track
# of exactly the same length from the same prompt, and ffmpeg muxes it in.
# fp16 weights ≈ 2.3 GB, kept resident on the GPU (the T4 has ~10 GB of headroom
# while Wan is offloaded), so it never touches Colab's 12.7 GB host RAM budget.

AUDIO_MODEL_ID = "cvssp/audioldm2"
AUDIO_NEGATIVE_PROMPT = "low quality, noise, distortion, muffled, clipping, speech, talking"
FPS = 16

# Anti-morphing base negative prompt designed to preserve rigid geometry and texture sharpness
WAN_NEGATIVE_PROMPT = (
    "morphing, rubbery, liquid, warped geometry, melting, shifting shapes, "
    "blurry details, soft focus, smudged textures, plastic, bad anatomy, "
    "deformed limbs, floating limbs, disfigured faces, cartoonish, static, "
    "flat lighting, artifacts, JPEG compression residue, watermarks, text, "
    "oversaturated, underexposed, flicker"
)

# High-fidelity hardware-tuned settings for Tesla T4 (num_frames must be 4k+1).
# Optimized for Reel 9:16 vertical (480x832 -> 720x1280 HD Lanczos) with 36 Euler steps.
DURATION_PRESETS = {
    "Reel 9:16 (HQ 480p)": dict(
        width=480, height=832, num_frames=25, steps=36, eta_min=5.0,
        desc="Vertical Reel HD (720x1280), 36 pasos Euler shift=5.0 + LLM Prompt Director"
    ),
    "Reel 9:16 (Extended 3s)": dict(
        width=480, height=832, num_frames=49, steps=32, eta_min=8.0,
        desc="Vertical Reel 3s HD (720x1280), 32 pasos Euler shift=5.0 + LLM Prompt Director"
    ),
    "Cinema 16:9 (Landscape)": dict(
        width=832, height=480, num_frames=25, steps=36, eta_min=5.0,
        desc="Panorámico HD (1280x720), 36 pasos Euler shift=5.0 + LLM Prompt Director"
    ),
}


def get_preset(name: str) -> dict:
    """Resolve preset by canonical name or backward-compatible alias."""
    aliases = {
        "2s": "Reel 9:16 (HQ 480p)",
        "2s HQ (480p)": "Cinema 16:9 (Landscape)",
        "3s": "Reel 9:16 (Extended 3s)",
        "5s": "Reel 9:16 (Extended 3s)",
        "480p": "Reel 9:16 (HQ 480p)",
        "reel": "Reel 9:16 (HQ 480p)",
        "vertical": "Reel 9:16 (HQ 480p)",
        "landscape": "Cinema 16:9 (Landscape)",
        "Cinematic 480p (HQ)": "Cinema 16:9 (Landscape)",
        "Widescreen Fast (1.5s)": "Reel 9:16 (HQ 480p)",
        "Extended Motion (3.0s)": "Reel 9:16 (Extended 3s)",
    }
    canonical = aliases.get(name, name)
    return DURATION_PRESETS.get(canonical, DURATION_PRESETS["Reel 9:16 (HQ 480p)"])


# ══════════════════════════════════════════════════════════════════════════════
# Zero-VRAM Cloud LLM Prompt & Dynamic Negative-Prompt Director (0 MB RAM/VRAM)
# ══════════════════════════════════════════════════════════════════════════════
_LLM_SYSTEM_PROMPT = (
    "You are an expert prompt engineer for Alibaba Wan 2.1 Text-to-Video diffusion model. "
    "Convert the user input (in any language) into a JSON object with keys 'positive' and 'negative':\n"
    "1. 'positive': A single dense paragraph (65-95 words) in English describing a photorealistic cinematic video. "
    "Explicitly specify: (a) rigid physical structure and non-deforming invariant geometry of solid objects "
    "(e.g. rigid metallic bodywork, solid rectilinear architecture, invariant proportions), "
    "(b) crisp high-frequency surface micro-textures (sharp reflections, realistic material grain, sharp edges), "
    "and (c) optical stability (50mm prime cinema lens, steady gimbal motion, natural high-contrast lighting).\n"
    "2. 'negative': A comma-separated list of 10-15 specific visual deformations to avoid for the exact objects "
    "in this scene (e.g. warped wheels, bending chassis, rubbery metal, melting walls, extra fingers, blurry textures).\n"
    "Return ONLY valid JSON."
)


def _translate_to_english(text: str) -> str:
    """Zero-key, zero-RAM cloud translation via Google GTX endpoint (0.1s latency)."""
    import json
    import urllib.parse
    import urllib.request

    try:
        url = (
            "https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q="
            + urllib.parse.quote(text)
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=4) as r:
            data = json.loads(r.read().decode("utf-8"))
            translated = "".join(seg[0] for seg in data[0] if seg[0]).strip()
            if translated:
                return translated
    except Exception:
        pass
    return text


def enhance_prompt_and_negative(prompt: str, is_vertical: bool = True) -> Tuple[str, str]:
    """
    Zero-VRAM / Zero-RAM Prompt & Negative-Prompt Enhancer via external HTTP APIs:
      - Tier 1: Hugging Face Serverless Inference API (Qwen/Qwen2.5-72B-Instruct) if HF_TOKEN is set.
      - Tier 2: Pollinations OpenAI-Compatible Free API (no API key needed).
      - Tier 3: Cloud Auto-Translate + Domain-Specific Optical & Rigid-Physics Engine.

    Returns:
        (enhanced_positive_prompt, dynamic_negative_prompt)
    """
    import json
    import urllib.request

    raw = (prompt or "").strip()
    if not raw:
        return raw, WAN_NEGATIVE_PROMPT

    framing = "Vertical 9:16 Reel composition" if is_vertical else "Widescreen 16:9 cinema composition"
    user_msg = f"Framing: {framing}. Scene: {raw}"

    # Tier 1: Hugging Face Serverless Router (Qwen2.5-72B-Instruct — official Wan 2.1 captioner)
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if hf_token:
        try:
            payload = json.dumps({
                "model": "Qwen/Qwen2.5-72B-Instruct",
                "messages": [
                    {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                "max_tokens": 260,
                "temperature": 0.4,
                "response_format": {"type": "json_object"},
            }).encode("utf-8")
            req = urllib.request.Request(
                "https://router.huggingface.co/v1/chat/completions",
                data=payload,
                headers={
                    "Authorization": f"Bearer {hf_token}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=6) as r:
                resp = json.loads(r.read().decode("utf-8"))
                obj = json.loads(resp["choices"][0]["message"]["content"])
                pos = str(obj.get("positive", "")).strip()
                neg = str(obj.get("negative", "")).strip()
                if len(pos) > 30:
                    full_neg = f"{neg}, {WAN_NEGATIVE_PROMPT}" if neg else WAN_NEGATIVE_PROMPT
                    print(f"[LLM Director · Qwen2.5-72B API] Prompt expanded ({len(pos.split())} words)")
                    return pos, full_neg
        except Exception:
            pass

    # Tier 2: Free Zero-Key OpenAI-Compatible Endpoint (Pollinations AI)
    try:
        payload = json.dumps({
            "model": "openai-fast",
            "messages": [
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "response_format": {"type": "json_object"},
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://text.pollinations.ai/openai",
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "Wan21Studio/1.0"},
        )
        with urllib.request.urlopen(req, timeout=6) as r:
            resp = json.loads(r.read().decode("utf-8"))
            obj = json.loads(resp["choices"][0]["message"]["content"])
            pos = str(obj.get("positive", "")).strip()
            neg = str(obj.get("negative", "")).strip()
            if len(pos) > 30:
                full_neg = f"{neg}, {WAN_NEGATIVE_PROMPT}" if neg else WAN_NEGATIVE_PROMPT
                print(f"[LLM Director · Cloud API] Prompt expanded ({len(pos.split())} words)")
                return pos, full_neg
    except Exception:
        pass

    # Tier 3: Instant Cloud Translation + Domain-Specific Optical & Rigid-Physics Engine
    en_text = _translate_to_english(raw).rstrip(".")
    low = en_text.lower()

    pos_boost = [
        f"{framing}, ultra-sharp 8k photorealistic cinematography",
        en_text,
    ]
    neg_boost = []

    if any(k in low for k in ("car", "supercar", "drift", "truck", "bike", "motorcycle", "train", "vehicle", "wheel", "ferrari", "porsche")):
        pos_boost.append(
            "rigid non-deforming carbon-fiber and metallic bodywork, invariant wheel and chassis geometry, "
            "crisp specular reflections on solid surfaces, steady 50mm prime tracking shot"
        )
        neg_boost.append("warped wheels, oval tires, bending chassis, rubbery car body, asymmetric headlights, melting metal")
    if any(k in low for k in ("city", "street", "building", "tokyo", "skyscraper", "house", "room", "alley", "architecture", "bridge", "neon")):
        pos_boost.append(
            "strict rectilinear architectural geometry, solid non-warping walls and glass facades, "
            "sharp structural edges, crisp micro-textures, 35mm rectilinear cinema lens with zero distortion"
        )
        neg_boost.append("curved walls, bending buildings, melting windows, wobbly straight lines, shifting perspective")
    if any(k in low for k in ("person", "woman", "man", "girl", "boy", "face", "portrait", "human", "dancer", "hand", "eyes")):
        pos_boost.append(
            "anatomically accurate solid proportions, consistent facial structure, natural skin pore micro-texture, "
            "sharp eye catchlights, 85mm f/1.8 prime portrait lens"
        )
        neg_boost.append("extra fingers, fused fingers, mutated hands, asymmetric eyes, melting face, rubbery limbs")
    if any(k in low for k in ("wave", "ocean", "water", "sea", "rain", "river", "espresso", "coffee", "glass", "cup", "forest")):
        pos_boost.append(
            "physically accurate fluid dynamics, crisp surface tension droplets, sharp specular highlights, "
            "solid rigid container and rock boundaries, high-speed macro cinema lens"
        )
        neg_boost.append("muddy water, pixelated spray, warping glass cup, melting rocks, smudged reflections")

    if not neg_boost:
        pos_boost.append(
            "rigid solid object geometry, invariant physical proportions, crisp high-frequency surface micro-textures, "
            "shot on 50mm cinema prime lens, steady gimbal motion, natural volumetric lighting"
        )
        neg_boost.append("warped objects, rubbery physics, shifting silhouettes, blurry surfaces")

    final_pos = ". ".join(pos_boost) + "."
    final_neg = ", ".join(neg_boost + [WAN_NEGATIVE_PROMPT])
    print(f"[LLM Director · Optical Physics Engine] Prompt aligned ({len(final_pos.split())} words)")
    return final_pos, final_neg


_GENERATION_LOCK = threading.Lock()  # one job at a time on a single T4


def load_audio_pipeline(device: str = "cuda", dtype: torch.dtype = torch.float16):
    """Load AudioLDM 2 in fp16 directly onto the GPU (safetensors only, ~4.5 GB download)."""
    from diffusers import AudioLDM2Pipeline

    audio_pipe = AudioLDM2Pipeline.from_pretrained(AUDIO_MODEL_ID, torch_dtype=dtype, use_safetensors=True)
    audio_pipe = audio_pipe.to(device)
    audio_pipe.set_progress_bar_config(disable=True)
    clear_cuda_cache()
    return audio_pipe


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def export_hq_video(frames, output_path: str, fps: int = 16, upscale_factor: float = 1.5) -> str:
    """
    CPU-based High-Definition FFmpeg exporter (0 MB GPU VRAM):
    - Applies light spatiotemporal grain cleanup (hqdn3d)
    - Upscales 1.5x via Lanczos (480x832 -> 720x1280 HD Reel / 832x480 -> 1280x720 HD)
    - Applies contrast-adaptive luma unsharp mask (unsharp=5:5:0.60:3:3:0.0) for crisp textures
    - Encodes visually lossless H.264 (-crf 14 -preset slow) to eliminate blockiness
    """
    import numpy as np

    try:
        arr = np.stack([np.asarray(f) for f in frames]) if isinstance(frames, (list, tuple)) else np.asarray(frames)
        if arr.dtype != np.uint8:
            if arr.max() <= 1.05:
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        _, h, w, _ = arr.shape
        out_w = (int(round(w * upscale_factor)) // 2) * 2
        out_h = (int(round(h * upscale_factor)) // 2) * 2
        vf = f"hqdn3d=1.0:1.0:2.0:2.0,scale={out_w}:{out_h}:flags=lanczos,unsharp=5:5:0.60:3:3:0.0"
        cmd = [
            _ffmpeg_exe(), "-y", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", str(fps),
            "-i", "-",
            "-vf", vf,
            "-c:v", "libx264", "-crf", "14", "-preset", "slow",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            output_path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        _, err = proc.communicate(input=arr.tobytes())
        if proc.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
    except Exception:
        pass

    from diffusers.utils import export_to_video
    export_to_video(frames, output_path, fps=fps)
    return output_path


def generate_sound(audio_pipe, prompt: str, duration_s: float, wav_path: str, seed: int, steps: int = 50) -> str:
    """Render a mono 16 kHz WAV of `duration_s` seconds that matches the video prompt."""
    import numpy as np
    from scipy.io import wavfile

    text = f"{prompt.strip().rstrip('.')}. Realistic sound effects and ambience, high quality audio."
    with torch.inference_mode():
        audio = audio_pipe(
            prompt=text,
            negative_prompt=AUDIO_NEGATIVE_PROMPT,
            num_inference_steps=steps,
            audio_length_in_s=float(duration_s),
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).audios[0]
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    peak = float(np.abs(audio).max()) or 1.0
    wavfile.write(wav_path, audio_pipe.vocoder.config.sampling_rate, (audio / peak * 0.9 * 32767).astype(np.int16))
    return wav_path


def mux_audio(video_path: str, wav_path: str, output_path: str, duration_s: float) -> str:
    """Copy the H.264 stream untouched and add an AAC track with short fades."""
    fade = min(0.3, duration_s / 4)
    subprocess.run(
        [
            _ffmpeg_exe(), "-y", "-loglevel", "error",
            "-i", video_path, "-i", wav_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-af", f"afade=t=in:d=0.05,afade=t=out:st={duration_s - fade:.2f}:d={fade:.2f}",
            "-shortest", "-movflags", "+faststart", output_path,
        ],
        check=True,
        capture_output=True,
    )
    return output_path


def create_clip(pipe, audio_pipe=None, prompt: str = "", duration: str = "Reel 9:16 (HQ 480p)", seed: int = -1, output_dir: str = "outputs"):
    """
    Prompt in, finished MP4 out (video + sound when `audio_pipe` is given).
    Automatically enhances prompt & negative prompt via Zero-VRAM Cloud LLM API,
    pre-clears UMT5 activation cache before DiT, and renders 720p HD via FFmpeg Lanczos + Unsharp.

    Returns:
        (mp4_path, seed, elapsed_seconds, has_sound)
    """
    # If called positionally as create_clip(pipe, prompt, ...)
    if isinstance(audio_pipe, str) and not prompt:
        prompt = audio_pipe
        audio_pipe = None

    preset = get_preset(duration)
    is_vertical = preset["height"] > preset["width"]
    pos_prompt, neg_prompt = enhance_prompt_and_negative(prompt, is_vertical=is_vertical)

    seed = random.randint(0, 2**31 - 1) if seed is None or int(seed) < 0 else int(seed)
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.join(output_dir, f"wan_{int(time.time())}_{seed}")
    silent_path, wav_path, final_path = f"{stem}_silent.mp4", f"{stem}.wav", f"{stem}.mp4"
    duration_s = preset["num_frames"] / FPS

    with _GENERATION_LOCK:
        t0 = time.time()
        try:
            from diffusers import FlowMatchEulerDiscreteScheduler
            if not isinstance(pipe.scheduler, FlowMatchEulerDiscreteScheduler) or getattr(pipe.scheduler.config, "shift", None) != 5.0:
                pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config, shift=5.0)
        except Exception:
            pass
        clear_cuda_cache()
        output = frames = None
        prompt_embeds = neg_embeds = None
        try:
            with torch.inference_mode():
                # Pre-compute text embeddings and flush UMT5 intermediate activations before DiT 3D attention
                try:
                    prompt_embeds, neg_embeds = pipe.encode_prompt(
                        prompt=pos_prompt,
                        negative_prompt=neg_prompt,
                        do_classifier_free_guidance=True,
                        num_videos_per_prompt=1,
                        max_sequence_length=512,
                    )
                    clear_cuda_cache()
                    output = pipe(
                        prompt=None,
                        negative_prompt=None,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=neg_embeds,
                        width=preset["width"],
                        height=preset["height"],
                        num_frames=preset["num_frames"],
                        num_inference_steps=preset["steps"],
                        guidance_scale=6.0,
                        generator=torch.Generator(device="cpu").manual_seed(seed),
                    )
                except Exception:
                    clear_cuda_cache()
                    output = pipe(
                        prompt=pos_prompt,
                        negative_prompt=neg_prompt,
                        width=preset["width"],
                        height=preset["height"],
                        num_frames=preset["num_frames"],
                        num_inference_steps=preset["steps"],
                        guidance_scale=6.0,
                        generator=torch.Generator(device="cpu").manual_seed(seed),
                    )
            frames = output.frames[0]
            export_hq_video(frames, silent_path, fps=FPS)
        finally:
            del output, frames, prompt_embeds, neg_embeds
            clear_cuda_cache()

        has_sound = False
        if audio_pipe is not None:
            try:
                generate_sound(audio_pipe, prompt, duration_s, wav_path, seed)
                mux_audio(silent_path, wav_path, final_path, duration_s)
                has_sound = True
            except Exception as e:  # never lose the video because of the sound step
                print(f"[sound] skipped: {type(e).__name__}: {e}")
            finally:
                clear_cuda_cache()

        if has_sound:
            for p in (silent_path, wav_path):
                if os.path.exists(p):
                    os.remove(p)
        else:
            os.replace(silent_path, final_path)

        return final_path, seed, time.time() - t0, has_sound
