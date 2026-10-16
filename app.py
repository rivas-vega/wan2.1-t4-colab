"""
app.py - Clean, single-stage web interface for Wan 2.1 (4s-5s video) on a free Colab T4.

Workflow:
- Write a prompt, select 4s (65 frames) or 5s (81 frames), and click Create.
- While the 3D video is rendering, the stage displays a live progressive frame preview
  decoded directly from the active 3D video tensor (100% identical scene & framing).
- Click Stop anytime if you want to cancel early and try another prompt.
"""

import inspect
import queue
import threading

import gradio as gr

from pipeline_utils import (
    DURATION_PRESETS,
    create_clip,
    get_cached_pipeline,
    load_wan_t4_pipeline,
)

EXAMPLES = {
    "City Vlog": "A young content creator holding a camera at arm's length, walking through a sunny European street and smiling naturally",
    "Cafe Morning": "A guy in a knit sweater sitting by a large cafe window, opening his laptop and picking up a warm latte cup in natural daylight",
    "Tokyo Rain": "A matte-black sports car driving smoothly along a wet Tokyo street at night with clean streetlight reflections",
    "Coastal Wave": "Ocean waves rolling gently onto black volcanic sand at sunset under warm golden light",
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
.gradio-container { max-width: 780px !important; margin: 0 auto !important; padding: 48px 20px 40px !important; }
footer { display: none !important; }

#brand { display: flex; align-items: baseline; justify-content: space-between; margin: 0 0 20px 2px; }
#brand .left { display: flex; align-items: baseline; gap: 12px; }
#brand .name { font-size: 22px; font-weight: 600; letter-spacing: -0.02em; color: #f2f2f3; }
#brand .sub  { font-size: 14px; color: #8a8a92; }

#stage, #live-preview {
  max-width: 420px !important;
  margin: 0 auto 12px !important;
  border: 1px solid #232327 !important;
  border-radius: 20px !important;
  overflow: hidden;
  background: #111113 !important;
}
#stage video, #live-preview img { border-radius: 20px; }

#status, #status * { color: #a1a1aa !important; font-size: 13px !important; text-align: center; }
#status { min-height: 22px; margin: 10px 0 6px; }

#composer { border: 1px solid #2a2a2f !important; border-radius: 24px !important; background: #141416 !important;
            padding: 12px 12px 12px 18px !important; gap: 6px !important; }
#composer > * { border: none !important; background: transparent !important; }
#prompt textarea { background: transparent !important; border: none !important; box-shadow: none !important;
                   font-size: 16px !important; line-height: 1.5 !important; color: #f2f2f3 !important; padding: 4px 0 !important; resize: none; }
#prompt textarea::placeholder { color: #6e6e76 !important; }
#controls { align-items: center !important; justify-content: space-between !important; gap: 10px !important; }

#length { background: transparent !important; border: none !important; padding: 0 !important; flex: 0 1 auto !important; }
#length .wrap { gap: 6px !important; }
#length label { border-radius: 999px !important; padding: 6px 14px !important; font-size: 13px !important;
                background: #1b1b1e !important; border: 1px solid #2a2a2f !important; color: #b4b4ba !important; cursor: pointer; }
#length label.selected, #length label:has(input:checked) { background: #f2f2f3 !important; color: #0b0b0c !important; border-color: #f2f2f3 !important; }
#length input[type=radio] { display: none !important; }

#action-btns { flex: 0 0 auto !important; gap: 8px !important; justify-content: flex-end !important; }
#create { border-radius: 999px !important; min-width: 112px !important; max-width: 136px !important; font-weight: 600 !important; }
#stop { border-radius: 999px !important; min-width: 84px !important; max-width: 100px !important; font-weight: 500 !important;
        border: 1px solid #3f3f46 !important; color: #e4e4e7 !important; }

#hint, #hint * { color: #6e6e76 !important; font-size: 12.5px !important; text-align: center; }
#hint { margin-top: 8px; }

#chips { justify-content: center !important; flex-wrap: wrap !important; gap: 8px !important; margin-top: 16px; }
.chip { flex: 0 0 auto !important; min-width: 0 !important; width: auto !important; border-radius: 999px !important;
        border: 1px solid #2a2a2f !important; font-weight: 400 !important; padding: 6px 14px !important; }
.chip:hover { color: #f2f2f3 !important; }
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.neutral,
    neutral_hue=gr.themes.colors.neutral,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

_GRADIO6 = "css" in inspect.signature(gr.Blocks.launch).parameters


def _hint(length: str, has_sound: bool = False) -> str:
    preset = DURATION_PRESETS.get(length, {})
    minutes = preset.get("eta_min", 7.5)
    desc = preset.get("desc", "")
    sound_suffix = " · includes sound" if has_sound else ""
    return f"{desc} · Live preview at step 8 (~{minutes} min total on T4){sound_suffix}"


def build_demo(pipe=None, audio_pipe=None, output_dir: str = "outputs") -> gr.Blocks:
    has_sound = audio_pipe is not None
    default_length = "Reel 9:16 · 4s"
    state_ref = {
        "pipe": get_cached_pipeline(pipe),
        "cancel_event": threading.Event(),
    }

    def run(prompt_text: str, length_name: str):
        prompt_text = (prompt_text or "").strip()
        if not prompt_text:
            raise gr.Error("Describe the video you want to create.")

        state_ref["cancel_event"].clear()

        # 1. Auto-load pipeline if not yet loaded
        if get_cached_pipeline(state_ref["pipe"]) is None:
            q_load: queue.Queue = queue.Queue()
            load_res = {}

            def _loader():
                try:
                    load_res["pipe"] = load_wan_t4_pipeline(
                        quantization="fast_fp16",
                        status_cb=lambda m: q_load.put(m),
                    )
                except Exception as exc:
                    load_res["err"] = exc
                finally:
                    q_load.put(None)

            t_load = threading.Thread(target=_loader, daemon=True)
            t_load.start()
            while True:
                msg = q_load.get()
                if msg is None:
                    break
                yield gr.update(visible=True), gr.update(visible=False), msg, gr.update(visible=True)
            t_load.join()
            if "err" in load_res:
                raise gr.Error(f"Model initialization failed: {load_res['err']}")
            state_ref["pipe"] = load_res["pipe"]
        else:
            state_ref["pipe"] = get_cached_pipeline(state_ref["pipe"])

        # 2. Run 3D video generation with live frame preview from the active 3D trajectory
        q_gen: queue.Queue = queue.Queue()
        gen_res = {}

        def _worker():
            try:
                gen_res["out"] = create_clip(
                    state_ref["pipe"],
                    audio_pipe=audio_pipe,
                    prompt=prompt_text,
                    duration=length_name,
                    output_dir=output_dir,
                    progress_cb=lambda msg, prev_img=None: q_gen.put((msg, prev_img)),
                    cancel_event=state_ref["cancel_event"],
                )
            except Exception as exc:
                gen_res["err"] = exc
            finally:
                q_gen.put(None)

        t_gen = threading.Thread(target=_worker, daemon=True)
        t_gen.start()

        shown_preview = None
        while True:
            item = q_gen.get()
            if item is None:
                break
            msg, prev_img = item
            if prev_img and prev_img != shown_preview:
                shown_preview = prev_img
                yield (
                    gr.update(visible=False),
                    gr.update(value=prev_img, visible=True),
                    f"{msg} · Live frame from current video",
                    gr.update(visible=True),
                )
            else:
                yield (
                    gr.update(),
                    gr.update(),
                    msg,
                    gr.update(visible=True),
                )
        t_gen.join()

        if "err" in gen_res:
            err = gen_res["err"]
            if isinstance(err, InterruptedError):
                yield (
                    gr.update(visible=shown_preview is None),
                    gr.update(visible=shown_preview is not None),
                    "Generation stopped. Adjust your prompt or click Create for a new take.",
                    gr.update(visible=False),
                )
                return
            raise gr.Error(str(err))

        path, seed, secs, sound = gen_res["out"]
        summary = f"{secs / 60:.1f} min · seed {seed}" + (" · with sound" if sound else "")
        yield (
            gr.update(value=path, visible=True),
            gr.update(visible=False),
            summary,
            gr.update(visible=False),
        )

    def stop_generation():
        state_ref["cancel_event"].set()
        return "Stopping after current diffusion step..."

    blocks_kwargs = {} if _GRADIO6 else {"theme": THEME, "css": CSS}
    with gr.Blocks(title="Wan 2.1", **blocks_kwargs) as demo:
        gr.HTML(
            '<div id="brand">'
            '<div class="left"><span class="name">Wan 2.1</span><span class="sub">Open Video Synthesis on Free T4</span></div>'
            "</div>"
        )

        video = gr.Video(show_label=False, autoplay=True, loop=True, interactive=False, visible=True, elem_id="stage")
        live_preview = gr.Image(
            show_label=False,
            interactive=False,
            visible=False,
             container=False,
            elem_id="live-preview",
        )
        status = gr.Markdown("", elem_id="status")

        with gr.Group(elem_id="composer"):
            prompt = gr.Textbox(
                show_label=False,
                container=False,
                lines=2,
                max_lines=6,
                placeholder="Describe a scene (subject, lighting, movement)...",
                elem_id="prompt",
            )
            with gr.Row(elem_id="controls"):
                length = gr.Radio(
                    choices=list(DURATION_PRESETS.keys()),
                    value=default_length,
                    show_label=False,
                    container=False,
                    elem_id="length",
                )
                with gr.Row(elem_id="action-btns"):
                    stop_btn = gr.Button("Stop", variant="secondary", scale=0, visible=False, elem_id="stop")
                    create = gr.Button("Create", variant="primary", scale=0, elem_id="create")

        hint = gr.Markdown(_hint(default_length, has_sound), elem_id="hint")

        with gr.Row(elem_id="chips"):
            for label, text in EXAMPLES.items():
                chip = gr.Button(label, size="sm", variant="secondary", elem_classes="chip")
                chip.click(lambda t=text: t, inputs=None, outputs=prompt, queue=False)

        length.change(lambda l: _hint(l, has_sound), inputs=length, outputs=hint, queue=False)
        create.click(
            run,
            inputs=[prompt, length],
            outputs=[video, live_preview, status, stop_btn],
            concurrency_limit=1,
        )
        stop_btn.click(stop_generation, inputs=None, outputs=status, queue=False)

    return demo


def launch(demo: gr.Blocks, share: bool = True, **kwargs):
    launch_kwargs = {"theme": THEME, "css": CSS} if _GRADIO6 else {}
    launch_kwargs.update(kwargs)
    return demo.queue(default_concurrency_limit=1, max_size=8).launch(
        share=share, show_error=True, **launch_kwargs
    )


if __name__ == "__main__":
    launch(build_demo(pipe=None), inbrowser=True)
