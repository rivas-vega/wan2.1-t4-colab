"""
app.py - Hardware-Optimized Gradio Web Interface for Wan 2.1 on Google Colab (T4 16GB).

Run locally or in Google Colab:
    python app.py
"""

import os
import random
import gradio as gr

# Restrict Linux glibc allocator heap arena hoarding in container environments (Colab)
os.environ["MALLOC_TRIM_THRESHOLD_"] = "65536"

# Prevent Colab torchao import conflict
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

from pipeline_utils import (
    load_wan_t4_pipeline,
    generate_video,
    get_gpu_memory_summary,
    clear_cuda_cache,
    calculate_spatiotemporal_tokens,
    estimate_generation_time_seconds,
)

# Global pipeline instance (loaded once)
PIPE = None


def get_or_load_pipeline():
    global PIPE
    if PIPE is None:
        PIPE = load_wan_t4_pipeline()
    return PIPE


RESOLUTION_MAP = {
    "624x352 (Widescreen 16:9 - Balanced, ~3 min)": (624, 352),
    "512x320 (Widescreen 16:9 - Fast Draft, ~1.5 min)": (512, 320),
    "832x480 (Cinematic 16:9 - High Quality, ~8-14 min)": (832, 480),
    "320x512 (Portrait 9:16 - Fast Draft, ~1.5 min)": (320, 512),
    "480x832 (Portrait 9:16 - High Quality, ~8-14 min)": (480, 832),
    "512x512 (Square 1:1 - Fast, ~2.5 min)": (512, 512),
    "624x624 (Square 1:1 - Standard, ~6 min)": (624, 624),
}


def compute_metrics_markdown(resolution: str, num_frames: int, steps: int, guidance: float) -> str:
    w, h = RESOLUTION_MAP.get(resolution, (624, 352))
    tokens, lt, lh, lw = calculate_spatiotemporal_tokens(w, h, int(num_frames))
    est_sec = estimate_generation_time_seconds(w, h, int(num_frames), int(steps), float(guidance), is_fp16=True)
    est_min = est_sec / 60.0
    passes = int(steps) * (2 if float(guidance) > 1.0 else 1)

    return (
        f"⏱️ **Estimated T4 Runtime:** `~{est_min:.1f} min` ({est_sec:.0f}s) | "
        f"🧩 **3D Tokens:** `{tokens:,}` latents `[{lt}x{lh}x{lw}]` | "
        f"🔁 **Forward Passes:** `{passes}` ({steps} st {'x 2 (CFG)' if passes > steps else ''})"
    )


def apply_preset(preset_name: str):
    if "Fast Draft" in preset_name:
        return "512x320 (Widescreen 16:9 - Fast Draft, ~1.5 min)", 17, 15, 5.0
    elif "Balanced" in preset_name:
        return "624x352 (Widescreen 16:9 - Balanced, ~3 min)", 25, 16, 5.0
    elif "Cinematic" in preset_name or "High" in preset_name:
        return "832x480 (Cinematic 16:9 - High Quality, ~8-14 min)", 33, 20, 5.0
    return "624x352 (Widescreen 16:9 - Balanced, ~3 min)", 25, 16, 5.0


def infer_video(
    prompt: str,
    negative_prompt: str,
    resolution: str,
    num_frames: int,
    steps: int,
    guidance_scale: float,
    seed: int,
):
    """Callback for the Gradio interface to trigger video synthesis without memory leaks."""
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a text prompt.")

    width, height = RESOLUTION_MAP.get(resolution, (624, 352))
    actual_seed = random.randint(0, 2**31 - 1) if seed == -1 else int(seed)

    pipe = get_or_load_pipeline()
    output_filename = f"wan2_1_output_{actual_seed}.mp4"

    try:
        video_path = generate_video(
            pipe=pipe,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=int(num_frames),
            steps=int(steps),
            guidance_scale=float(guidance_scale),
            seed=actual_seed,
            output_path=output_filename,
            fps=16,
            callback=None,
        )
        vram_info = get_gpu_memory_summary()
        return video_path, f"✓ Generation complete! Seed: {actual_seed} | {vram_info}"
    except Exception as e:
        clear_cuda_cache()
        error_msg = f"Generation error: {str(e)}"
        print(f"[Error] {error_msg}")
        return None, error_msg
    finally:
        clear_cuda_cache()


def create_ui():
    """Build the Gradio Blocks application."""
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
    )

    examples = [
        [
            "A sleek cyberpunk racing car drifting around a sharp neon-lit curve at night, tire smoke, reflections in wet asphalt, cinematic 4k.",
            "Bright, overexposed, static, low quality, distorted, blurry, deformed",
            "624x352 (Widescreen 16:9 - Balanced, ~3 min)",
            25,
            16,
            5.0,
            42,
        ],
        [
            "Cinematic drone shot flying through a misty Nordic pine forest during sunrise, golden light rays piercing fog.",
            "Bright, overexposed, static, low quality, distorted, blurry, deformed",
            "512x320 (Widescreen 16:9 - Fast Draft, ~1.5 min)",
            17,
            15,
            5.0,
            1337,
        ],
        [
            "Intricate macro shot of an antique mechanical wristwatch mechanism ticking, precision golden gears moving smoothly.",
            "Bright, overexposed, static, low quality, distorted, blurry, deformed",
            "512x512 (Square 1:1 - Fast, ~2.5 min)",
            25,
            16,
            5.0,
            2026,
        ],
    ]

    with gr.Blocks(theme=theme, title="Wan 2.1 Video Generator (Colab T4)") as demo:
        gr.Markdown(
            """
            # 🎬 Wan 2.1 Video Generation on Google Colab (Free T4 16GB)
            High-performance video generation using **Alibaba's Wan 2.1 (T2V-1.3B)**.  
            Optimized for **NVIDIA Tesla T4 Turing Tensor Cores (sm_75)** with **Model CPU Offload** & **3D VAE Tiling**.
            """
        )

        with gr.Row():
            with gr.Column(scale=5):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the video you want to generate...",
                    lines=3,
                    value="A sleek sports car drifting around a sharp mountain hairpin turn, tire smoke, sunset lighting, high speed camera.",
                )
                with gr.Accordion("Negative Prompt & Aesthetic Filters", open=False):
                    neg_prompt_input = gr.Textbox(
                        label="Negative Prompt",
                        lines=2,
                        value="Bright, overexposed, static, low quality, distorted, blurry, deformed, flickering",
                    )

                with gr.Row():
                    preset_dropdown = gr.Dropdown(
                        label="⚡ Speed / Quality Preset",
                        choices=[
                            "⚡ Fast Draft (~1.5 min) [512x320, 17f, 15st]",
                            "⚡ Balanced [Recommended] (~3 min) [624x352, 25f, 16st]",
                            "🎬 Cinematic Standard (~8 min) [832x480, 25f, 18st]",
                            "💎 High Quality (~14 min) [832x480, 33f, 20st]",
                        ],
                        value="⚡ Balanced [Recommended] (~3 min) [624x352, 25f, 16st]",
                    )

                with gr.Row():
                    res_dropdown = gr.Dropdown(
                        label="Resolution",
                        choices=list(RESOLUTION_MAP.keys()),
                        value="624x352 (Widescreen 16:9 - Balanced, ~3 min)",
                    )
                    frames_slider = gr.Slider(
                        label="Frames (25 ≈ 1.5s, 33 ≈ 2s @ 16fps)",
                        minimum=17,
                        maximum=49,
                        step=8,
                        value=25,
                    )

                with gr.Row():
                    steps_slider = gr.Slider(
                        label="Inference Steps (14-18 optimal with UniPC)",
                        minimum=12,
                        maximum=30,
                        step=1,
                        value=16,
                    )
                    guidance_slider = gr.Slider(
                        label="Guidance Scale (CFG)",
                        minimum=1.0,
                        maximum=8.0,
                        step=0.5,
                        value=5.0,
                    )

                metrics_display = gr.Markdown(
                    value=compute_metrics_markdown(
                        "624x352 (Widescreen 16:9 - Balanced, ~3 min)",
                        25,
                        16,
                        5.0,
                    )
                )

                with gr.Row():
                    seed_input = gr.Number(
                        label="Seed (-1 for random)",
                        value=-1,
                        precision=0,
                    )
                    random_seed_btn = gr.Button("🎲 Random Seed", scale=1)

                random_seed_btn.click(lambda: -1, outputs=[seed_input])

                generate_btn = gr.Button("🚀 Generate Video", variant="primary", size="lg")

            with gr.Column(scale=5):
                video_output = gr.Video(label="Generated Video", autoplay=True, loop=True)
                status_output = gr.Markdown("Ready. Model will load into GPU on first run.")

        # Preset event handlers
        preset_dropdown.change(
            fn=apply_preset,
            inputs=[preset_dropdown],
            outputs=[res_dropdown, frames_slider, steps_slider, guidance_slider],
        ).then(
            fn=compute_metrics_markdown,
            inputs=[res_dropdown, frames_slider, steps_slider, guidance_slider],
            outputs=[metrics_display],
        )

        for component in [res_dropdown, frames_slider, steps_slider, guidance_slider]:
            component.change(
                fn=compute_metrics_markdown,
                inputs=[res_dropdown, frames_slider, steps_slider, guidance_slider],
                outputs=[metrics_display],
            )

        gr.Examples(
            examples=examples,
            inputs=[
                prompt_input,
                neg_prompt_input,
                res_dropdown,
                frames_slider,
                steps_slider,
                guidance_slider,
                seed_input,
            ],
            label="Curated Example Prompts",
        )

        gr.Markdown(
            """
            ---
            ### 📐 Why Does Video Resolution Dictate Generation Time?
            - **Spatial & Temporal Latent Downsampling:** Wan 2.1 compresses space by $8\\times$ and time by $4\\times$.
            - **Quadratic 3D Attention:** Self-attention scales as $\\mathcal{O}(S^2 \\cdot d)$ where $S = \\left(\\frac{F-1}{4} + 1\\right) \\times \\frac{H}{8} \\times \\frac{W}{8}$.
            - **Token Comparison:** $512\\times 320$ (17 frames) has **12,800 tokens** vs $832\\times 480$ (33 frames) with **56,160 tokens** ($4.4\\times$ more tokens $\\to \\mathbf{19\\times}$ more attention FLOPs!).
            - **Native FP16:** Runs natively on T4 Turing Tensor Cores without dynamic register dequantization, saving ~40% latency over 4-bit.
            """
        )

        generate_btn.click(
            fn=infer_video,
            inputs=[
                prompt_input,
                neg_prompt_input,
                res_dropdown,
                frames_slider,
                steps_slider,
                guidance_slider,
                seed_input,
            ],
            outputs=[video_output, status_output],
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    # share=True creates a public gradio.live URL accessible from any device!
    demo.queue(default_concurrency_limit=1).launch(share=True, inbrowser=True)
