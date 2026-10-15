"""
app.py - Minimal web app for Wan 2.1 text-to-video (+ sound) on a free Colab T4.

Prompt in, video with sound out. All generation settings are fixed to the tuned
presets in pipeline_utils.DURATION_PRESETS; the only choice left to the user is length.

Run locally:
    python app.py
From the Colab notebook:
    from app import build_demo, launch
    launch(build_demo(pipe, audio_pipe))
"""

import inspect

import gradio as gr

from pipeline_utils import DURATION_PRESETS, create_clip

EXAMPLES = {
    "Drift at sunset": "A red vintage sports car drifting around a mountain hairpin at sunset, tire smoke, cinematic tracking shot",
    "Stormy coast": "Huge waves crashing against black volcanic rocks during a storm, sea spray, slow motion, moody light",
    "Neon rain": "Rain falling on a neon-lit Tokyo street at night, reflections on wet asphalt, people with umbrellas",
    "Forest stream": "A clear stream flowing over mossy rocks in a misty pine forest at dawn, soft light rays",
}

CSS = """
:root, .dark {
  --body-background-fill: #0b0b0c;
  --background-fill-primary: #0b0b0c;
  --background-fill-secondary: #141416;
  --block-background-fill: #141416;
  --block-border-color: #232327;
  --border-color-primary: #232327;
  --input-background-fill: #141416;
  --body-text-color: #f2f2f3;
  --body-text-color-subdued: #8a8a92;
  --block-label-text-color: #8a8a92;
  --button-primary-background-fill: #f2f2f3;
  --button-primary-background-fill-hover: #ffffff;
  --button-primary-text-color: #0b0b0c;
  --button-primary-text-color-hover: #0b0b0c;
  --button-secondary-background-fill: transparent;
  --button-secondary-background-fill-hover: #1b1b1e;
  --button-secondary-text-color: #b4b4ba;
  --color-accent: #f2f2f3;
  --color-accent-soft: #232327;
  --shadow-drop: none;
  --shadow-drop-lg: none;
}
body, gradio-app, .gradio-container { background: #0b0b0c !important; }
.gradio-container { max-width: 780px !important; margin: 0 auto !important; padding: 56px 20px 40px !important; }
footer { display: none !important; }

#brand { display: flex; align-items: baseline; gap: 12px; margin: 0 0 22px 2px; }
#brand .name { font-size: 22px; font-weight: 600; letter-spacing: -0.02em; color: #f2f2f3; }
#brand .sub  { font-size: 14px; color: #8a8a92; }

#stage { border: 1px solid #232327 !important; border-radius: 20px !important; overflow: hidden; background: #111113 !important; }
#stage video { border-radius: 20px; }

#status, #status * { color: #8a8a92 !important; font-size: 13px !important; text-align: center; }
#status { min-height: 22px; margin: 10px 0 2px; }

#composer { border: 1px solid #2a2a2f !important; border-radius: 24px !important; background: #141416 !important;
            padding: 12px 12px 12px 18px !important; gap: 6px !important; }
#composer > * { border: none !important; background: transparent !important; }
#prompt textarea { background: transparent !important; border: none !important; box-shadow: none !important;
                   font-size: 16px !important; line-height: 1.5 !important; color: #f2f2f3 !important; padding: 4px 0 !important; resize: none; }
#prompt textarea::placeholder { color: #6e6e76 !important; }
#controls { align-items: center !important; justify-content: space-between !important; gap: 12px !important; }

#length { background: transparent !important; border: none !important; padding: 0 !important; flex: 0 1 auto !important; }
#length .wrap { gap: 6px !important; }
#length label { border-radius: 999px !important; padding: 6px 14px !important; font-size: 13px !important;
                background: #1b1b1e !important; border: 1px solid #2a2a2f !important; color: #b4b4ba !important; cursor: pointer; }
#length label.selected, #length label:has(input:checked) { background: #f2f2f3 !important; color: #0b0b0c !important; border-color: #f2f2f3 !important; }
#length input[type=radio] { display: none !important; }

#create { border-radius: 999px !important; min-width: 112px !important; max-width: 140px !important; font-weight: 600 !important; }

#hint, #hint * { color: #6e6e76 !important; font-size: 12.5px !important; text-align: center; }
#hint { margin-top: 8px; }

#chips { justify-content: center !important; flex-wrap: wrap !important; gap: 8px !important; margin-top: 18px; }
.chip { flex: 0 0 auto !important; min-width: 0 !important; width: auto !important; border-radius: 999px !important;
        border: 1px solid #2a2a2f !important; font-weight: 400 !important; padding: 6px 14px !important; }
.chip:hover { color: #f2f2f3 !important; }
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.neutral,
    neutral_hue=gr.themes.colors.neutral,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

# Gradio 6 moved theme/css from gr.Blocks() to launch(); support both.
_GRADIO6 = "css" in inspect.signature(gr.Blocks.launch).parameters


def _hint(length: str, has_sound: bool) -> str:
    minutes = DURATION_PRESETS[length]["eta_min"]
    return f"About {minutes} min on a free T4" + (" · includes sound" if has_sound else "")


def build_demo(pipe, audio_pipe=None, output_dir: str = "outputs") -> gr.Blocks:
    has_sound = audio_pipe is not None

    def run(prompt: str, length: str):
        prompt = (prompt or "").strip()
        if not prompt:
            raise gr.Error("Describe the video you want to create.")
        path, seed, secs, sound = create_clip(pipe, audio_pipe, prompt, length, output_dir=output_dir)
        return path, f"{secs / 60:.1f} min · seed {seed}" + (" · with sound" if sound else "")

    blocks_kwargs = {} if _GRADIO6 else {"theme": THEME, "css": CSS}
    with gr.Blocks(title="Wan 2.1", **blocks_kwargs) as demo:
        gr.HTML('<div id="brand"><span class="name">Wan 2.1</span><span class="sub">Text to video</span></div>')

        video = gr.Video(show_label=False, autoplay=True, loop=True, interactive=False, elem_id="stage")
        status = gr.Markdown("", elem_id="status")

        with gr.Group(elem_id="composer"):
            prompt = gr.Textbox(
                show_label=False, container=False, lines=2, max_lines=6,
                placeholder="Describe a scene…", elem_id="prompt",
            )
            with gr.Row(elem_id="controls"):
                length = gr.Radio(
                    choices=list(DURATION_PRESETS.keys()), value="2s",
                    show_label=False, container=False, elem_id="length",
                )
                create = gr.Button("Create", variant="primary", scale=0, elem_id="create")

        hint = gr.Markdown(_hint("2s", has_sound), elem_id="hint")

        with gr.Row(elem_id="chips"):
            for label, text in EXAMPLES.items():
                chip = gr.Button(label, size="sm", variant="secondary", elem_classes="chip")
                chip.click(lambda t=text: t, inputs=None, outputs=prompt, queue=False)

        length.change(lambda l: _hint(l, has_sound), inputs=length, outputs=hint, queue=False)
        create.click(run, inputs=[prompt, length], outputs=[video, status], concurrency_limit=1)

    return demo


def launch(demo: gr.Blocks, share: bool = True, **kwargs):
    launch_kwargs = {"theme": THEME, "css": CSS} if _GRADIO6 else {}
    launch_kwargs.update(kwargs)
    return demo.queue(default_concurrency_limit=1, max_size=8).launch(share=share, show_error=True, **launch_kwargs)


if __name__ == "__main__":
    from pipeline_utils import load_audio_pipeline, load_wan_t4_pipeline

    audio = None
    try:
        audio = load_audio_pipeline()
    except Exception as e:
        print(f"[sound] disabled: {type(e).__name__}: {e}")
    wan = load_wan_t4_pipeline()
    launch(build_demo(wan, audio), inbrowser=True)
