"""
pipeline_utils.py - Memory-Optimized Pipeline Loader for Wan 2.1 on 16GB GPUs (NVIDIA T4).

Enables running Alibaba's Wan 2.1 (T2V-1.3B) in free Google Colab instances by combining:
1. Model CPU Offloading: Sequentially loads UMT5 text encoder, 1.3B DiT, and VAE.
2. VAE Tiling & Slicing: Prevents high-resolution 3D spatiotemporal latent decoding spikes.
3. Quantization / FP16 precision: Drops VRAM footprint from >16 GB down to ~7.8 GB peak.
"""

import os
import gc
import torch
from typing import Optional, Tuple


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
    """Aggressive memory collection to prevent fragmentation on T4."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_wan_t4_pipeline(
    model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    quantization: str = "4bit",  # "4bit" (0.5 bytes NF4), "8bit" (1 byte INT8), or "fp16" (2 bytes)
    dtype: torch.dtype = torch.float16,
):
    """
    Load Wan 2.1 with memory optimizations tailored for a 16GB NVIDIA Tesla T4.

    Args:
        model_id: HuggingFace repository for official Diffusers weights.
        quantization:
            - '4bit': 0.5 bytes per parameter (NF4 via bitsandbytes). Lowest VRAM (~4-6 GB).
            - '8bit': 1 byte per parameter (INT8 via bitsandbytes). Balanced (~6-7 GB).
            - 'fp16': 2 bytes per parameter (native half-precision with CPU offload ~7.8 GB).
        dtype: Computation dtype (torch.float16 recommended for Turing T4 architecture).

    Returns:
        Configured and memory-offloaded WanPipeline ready for inference.
    """
    clear_cuda_cache()

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

    print(f"[Wan 2.1 Loader] Loading VAE (float32 with tiling)...")
    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
    )

    transformer = None
    if quantization == "4bit":
        print(f"[Wan 2.1 Loader] Applying 4-BIT QUANTIZATION (0.5 bytes/param, NF4) to 1.3B DiT...")
        from diffusers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
        transformer = WanTransformer3DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=bnb_config,
            torch_dtype=dtype,
        )
    elif quantization == "8bit":
        print(f"[Wan 2.1 Loader] Applying 8-BIT QUANTIZATION (1 byte/param, INT8) to 1.3B DiT...")
        from diffusers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        transformer = WanTransformer3DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=bnb_config,
            torch_dtype=dtype,
        )
    else:
        print(f"[Wan 2.1 Loader] Loading in FP16 (2 bytes/param) with CPU offload...")

    print(f"[Wan 2.1 Loader] Assembling WanPipeline...")
    pipe_kwargs = {"vae": vae, "torch_dtype": dtype}
    if transformer is not None:
        pipe_kwargs["transformer"] = transformer

    pipe = WanPipeline.from_pretrained(model_id, **pipe_kwargs)

    # Enable Model CPU Offload: Moves sub-modules to GPU only when active
    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()

    # Enable VAE Tiling and Slicing (crucial to avoid OOM during 3D video decoding)
    if hasattr(pipe, "vae") and pipe.vae is not None:
        if hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        if hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()

    print(f"[Wan 2.1 Loader] Model loaded successfully ({quantization} mode). {get_gpu_memory_summary()}")
    return pipe


def generate_video(
    pipe,
    prompt: str,
    negative_prompt: str = "Bright, overexposed, static, low quality, distorted, blurry, deformed",
    width: int = 832,
    height: int = 480,
    num_frames: int = 33,
    steps: int = 25,
    guidance_scale: float = 5.0,
    seed: Optional[int] = None,
    output_path: str = "output_video.mp4",
    fps: int = 16,
) -> str:
    """
    Generate a video sequence and export to MP4.

    Args:
        pipe: Initialized WanPipeline.
        prompt: Positive text prompt.
        negative_prompt: Negative guidance prompt.
        width: Video width (must be divisible by 16, e.g. 832, 480, 624).
        height: Video height (must be divisible by 16, e.g. 480, 832, 624).
        num_frames: Frame count (33 is roughly 2s at 16fps, ideal for T4).
        steps: Denoising steps (20-30 recommended).
        guidance_scale: Classifier-free guidance scale (typically 5.0).
        seed: Random seed for reproducibility (-1 or None for random).
        output_path: Filepath where the MP4 video will be saved.
        fps: Playback framerate.

    Returns:
        Path to the saved MP4 video.
    """
    clear_cuda_cache()

    generator = None
    if seed is not None and seed >= 0:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    print(f"[Generation] Starting Wan 2.1 inference: '{prompt[:60]}...' ({width}x{height}, {num_frames} frames, {steps} steps)")

    with torch.inference_mode():
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

    frames = output.frames[0] if hasattr(output, "frames") else output.images[0]

    from diffusers.utils import export_to_video
    export_to_video(frames, output_path, fps=fps)
    clear_cuda_cache()

    print(f"[Generation] Video exported to {output_path}. {get_gpu_memory_summary()}")
    return output_path
