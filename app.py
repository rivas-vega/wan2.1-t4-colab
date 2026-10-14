"""
app.py - Interactive Gradio Web Interface for Wan 2.1 Video Generation on Google Colab (T4 16GB).

Run locally or directly in Google Colab:
    python app.py
"""

import os
import random
import gradio as gr
from pipeline_utils import load_wan_t4_pipeline, generate_video, get_gpu_memory_summary, clear_cuda_cache

# Global pipeline instance (loaded once)
PIPE = None


def get_or_load_pipeline():
    global PIPE
    if PIPE is None:
        PIPE = load_wan_t4_pipeline()
    return PIPE


def infer_video(
    prompt: str,
    negative_prompt: str,
    resolution: str,
    num_frames: int,
    steps: int,
    guidance_scale: float,
    seed: int,
    progress=gr.Progress(track_tqdm=True),
):
    """Callback for the Gradio interface to trigger video synthesis."""
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a text prompt.")

    # Parse resolution preset
    resolution_map = {
        "832x480 (Landscape 16:9)": (832, 480),
        "480x832 (Portrait 9:16)": (480, 832),
        "624x624 (Square 1:1)": (624, 624),
    }
    width, height = resolution_map.get(resolution, (832, 480))

    # Resolve seed
    actual_seed = random.randint(0, 2**31 - 1) if seed == -1 else int(seed)

    pipe = get_or_load_pipeline()

    output_filename = f"wan2_1_output_{actual_seed}.mp4"

    try:
        progress(0.1, desc="Encoding prompt and generating 3D spatiotemporal latents...")
        video_path = generate_video(
            pipe=pipe,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=actual_seed,
            output_path=output_filename,
            fps=16,
        )
        vram_info = get_gpu_memory_summary()
        return video_path, f"Generation complete! Seed: {actual_seed} | {vram_info}"
    except Exception as e:
        clear_cuda_cache()
        error_msg = f"Generation error: {str(e)}"
        print(f"[Error] {error_msg}")
        return None, error_msg


def create_ui():
    """Build the Gradio Blocks application."""
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
    )

    examples = [
        [
            "A majestic robotic tiger stalking through a rain-slicked cyberpunk street at night, neon reflections, cinematic lighting, 4k.",
            "static, blurry, deformed, low quality, overexposed",
            "832x480 (Landscape 16:9)",
            33,
            25,
            5.0,
            42,
        ],
        [
            "Cinematic drone shot flying through a lush misty Scandinavian pine forest during golden hour, cinematic color grading.",
            "static, blurry, deformed, low quality, overexposed",
            "832x480 (Landscape 16:9)",
            33,
            25,
            5.0,
            1337,
        ],
        [
            "Close-up of an intricate mechanical pocket watch ticking, gears moving in slow motion, macro lens, dramatic soft studio lighting.",
            "static, blurry, deformed, low quality, overexposed",
            "624x624 (Square 1:1)",
            33,
            25,
            5.0,
            2026,
        ],
    ]

    with gr.Blocks(theme=theme, title="Wan 2.1 Video Generator (Colab T4)") as demo:
        gr.Markdown(
            """
            # 🎬 Wan 2.1 Video Generation on Google Colab (Free T4 16GB)
            Generate high-quality videos using **Alibaba's Wan 2.1 (T2V-1.3B)**.
            Optimized with **Model CPU Offloading** and **3D VAE Tiling** to run comfortably within **15 GB VRAM**.
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
                with gr.Accordion("Negative Prompt & Quality Filters", open=False):
                    neg_prompt_input = gr.Textbox(
                        label="Negative Prompt",
                        lines=2,
                        value="Bright, overexposed, static, low quality, distorted, blurry, deformed, flickering",
                    )

                with gr.Row():
                    res_dropdown = gr.Dropdown(
                        label="Resolution",
                        choices=[
                            "832x480 (Landscape 16:9)",
                            "480x832 (Portrait 9:16)",
                            "624x624 (Square 1:1)",
                        ],
                        value="832x480 (Landscape 16:9)",
                    )
                    frames_slider = gr.Slider(
                        label="Frames (33 ≈ 2s @ 16fps)",
                        minimum=17,
                        maximum=49,
                        step=8,
                        value=33,
                    )

                with gr.Row():
                    steps_slider = gr.Slider(
                        label="Inference Steps",
                        minimum=15,
                        maximum=35,
                        step=1,
                        value=25,
                    )
                    guidance_slider = gr.Slider(
                        label="Guidance Scale",
                        minimum=3.0,
                        maximum=8.0,
                        step=0.5,
                        value=5.0,
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
                status_output = gr.Markdown("Ready. GPU memory will be allocated upon generation.")

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
            ### 💡 Technical Notes on Free Colab T4 (15GB VRAM) Execution
            - **Text Encoder Offload:** The 4.8B UMT5-XXL text encoder encodes the prompt and is immediately moved to CPU RAM, leaving VRAM dedicated to diffusion.
            - **1.3B Diffusion Transformer:** Operates in native FP16 with PyTorch Scaled Dot-Product Attention (SDPA).
            - **3D VAE Tiling:** Latents are decoded in temporal/spatial slices to prevent memory spikes during video rendering.
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
    demo.queue().launch(share=True, inbrowser=True)
