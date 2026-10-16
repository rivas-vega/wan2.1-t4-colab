"""
app.py - Meta-style progressive Video Studio for Wan 2.1 on a free Google Colab T4.

Workflow:
1. Compose scene & select format (Single Shot 9:16, 3-Shot Story 9:16, or Widescreen 16:9).
2. Preview Storyboard: unlocks 35mm keyframe previews step-by-step (~6s per shot on T4).
   If models are not yet loaded in VRAM, the studio initializes them automatically inside Gradio.
3. Render Full Video: unlocks once all shot previews are approved, streaming live step progress
   and assembling the final 720p HD 35mm MP4 Reel.
"""

import base64
import html
import inspect
import os
import queue
import random
import threading

import gradio as gr

from pipeline_utils import (
    DURATION_PRESETS,
    get_cached_pipeline,
    get_gpu_memory_summary,
    load_audio_pipeline,
    load_wan_t4_pipeline,
    plan_storyboard,
    render_shot_preview,
    render_storyboard_video,
)

EXAMPLES = {
    "Tokyo Rain": "A matte-black sports car driving smoothly along a wet Tokyo street at night under soft streetlights",
    "Coastal Wave": "Ocean waves rolling gently onto black volcanic sand at sunset with warm golden light",
    "Paris Portrait": "A young woman in a beige trench coat walking along a quiet Paris sidewalk in soft morning light",
    "Espresso Pour": "Warm espresso pouring slowly into a clear glass cup on a wooden cafe table in natural morning light",
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
  --button-secondary-background-fill: #18181b;
  --button-secondary-background-fill-hover: #222226;
  --button-secondary-text-color: #d4d4d8;
  --color-accent: #f2f2f3;
  --color-accent-soft: #232327;
  --shadow-drop: none;
  --shadow-drop-lg: none;
}
body, gradio-app, .gradio-container {
  background: #0b0b0c !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.gradio-container {
  max-width: 860px !important;
  margin: 0 auto !important;
  padding: 44px 20px 48px !important;
}
footer { display: none !important; }

/* Header & Engine Status Bar */
#studio-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin: 0 0 24px 0;
  padding-bottom: 16px;
  border-bottom: 1px solid #1c1c20;
}
.brand-wrap {
  display: flex;
  align-items: baseline;
  gap: 12px;
}
.brand-name {
  font-size: 20px;
  font-weight: 600;
  letter-spacing: -0.025em;
  color: #f2f2f3;
}
.brand-sub {
  font-size: 13px;
  color: #7e7e86;
  font-weight: 400;
}
.engine-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 5px 12px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 500;
  letter-spacing: 0.01em;
  background: #141416;
  border: 1px solid #26262b;
  color: #b4b4ba;
}
.engine-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #71717a;
}
.engine-dot.ready {
  background: #e4e4e7;
  box-shadow: 0 0 8px rgba(242, 242, 243, 0.45);
}

/* Composer */
#composer {
  border: 1px solid #26262b !important;
  border-radius: 20px !important;
  background: #141416 !important;
  padding: 14px 16px 14px 18px !important;
  gap: 10px !important;
}
#composer > * { border: none !important; background: transparent !important; }
#prompt textarea {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  font-size: 15.5px !important;
  line-height: 1.5 !important;
  color: #f2f2f3 !important;
  padding: 2px 0 !important;
  resize: none;
}
#prompt textarea::placeholder { color: #66666e !important; }
#controls {
  align-items: center !important;
  justify-content: space-between !important;
  gap: 12px !important;
  padding-top: 6px !important;
  border-top: 1px solid #1e1e22 !important;
}

#length { background: transparent !important; border: none !important; padding: 0 !important; flex: 1 1 auto !important; }
#length .wrap { gap: 6px !important; }
#length label {
  border-radius: 999px !important;
  padding: 6px 13px !important;
  font-size: 12.5px !important;
  font-weight: 500 !important;
  background: #1a1a1e !important;
  border: 1px solid #28282e !important;
  color: #a1a1aa !important;
  cursor: pointer;
  transition: all 0.15s ease;
}
#length label.selected, #length label:has(input:checked) {
  background: #f2f2f3 !important;
  color: #0b0b0c !important;
  border-color: #f2f2f3 !important;
}
#length input[type=radio] { display: none !important; }

#preview-btn {
  border-radius: 999px !important;
  min-width: 158px !important;
  max-width: 184px !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  padding: 8px 18px !important;
}

#hint, #hint * { color: #6e6e76 !important; font-size: 12px !important; text-align: center; }
#hint { margin-top: 8px; }

#chips { justify-content: center !important; flex-wrap: wrap !important; gap: 8px !important; margin-top: 12px; }
.chip {
  flex: 0 0 auto !important;
  min-width: 0 !important;
  width: auto !important;
  border-radius: 999px !important;
  border: 1px solid #242429 !important;
  background: #121214 !important;
  color: #a1a1aa !important;
  font-size: 12px !important;
  font-weight: 400 !important;
  padding: 5px 13px !important;
}
.chip:hover { color: #f2f2f3 !important; border-color: #3a3a42 !important; }

/* Status Line */
#status, #status * {
  color: #a1a1aa !important;
  font-size: 13px !important;
  text-align: center;
  font-weight: 400;
}
#status { min-height: 22px; margin: 16px 0 8px; }

/* Progressive Storyboard Section */
#storyboard-section {
  margin-top: 14px;
  padding: 18px;
  border: 1px solid #232328;
  border-radius: 20px;
  background: #111113;
}
.sb-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 14px;
}
.sb-title {
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #a1a1aa;
}
.sb-progress {
  font-size: 12px;
  color: #7e7e86;
}
.sb-grid {
  display: grid;
  gap: 12px;
  width: 100%;
}
.sb-grid.cols-1 {
  grid-template-columns: minmax(0, 320px);
  justify-content: center;
}
.sb-grid.cols-3 {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}
@media (max-width: 680px) {
  .sb-grid.cols-3 { grid-template-columns: 1fr; }
}
.sb-card {
  border: 1px solid #242429;
  border-radius: 14px;
  background: #161619;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  transition: border-color 0.2s ease;
}
.sb-card.unlocked {
  border-color: #383840;
}
.sb-card.active {
  border-color: #71717a;
}
.sb-frame {
  width: 100%;
  background: #0e0e10;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}
.sb-frame.vertical { aspect-ratio: 9 / 16; max-height: 340px; }
.sb-frame.horizontal { aspect-ratio: 16 / 9; max-height: 220px; }
.sb-frame img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.sb-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 20px;
  text-align: center;
  gap: 6px;
}
.sb-state-tag {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: #71717a;
  padding: 3px 9px;
  border-radius: 999px;
  border: 1px solid #27272c;
  background: #141417;
}
.sb-card.active .sb-state-tag {
  color: #e4e4e7;
  border-color: #52525b;
}
.sb-meta {
  padding: 10px 12px;
  border-top: 1px solid #202025;
  display: flex;
  flex-direction: column;
  gap: 3px;
}
.sb-shot-name {
  font-size: 12.5px;
  font-weight: 600;
  color: #e4e4e7;
}
.sb-shot-cam {
  font-size: 11.5px;
  color: #8a8a92;
}

#storyboard-actions {
  margin-top: 14px !important;
  justify-content: flex-end !important;
  gap: 10px !important;
}
#reroll-btn, #render-btn {
  border-radius: 999px !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  padding: 8px 18px !important;
  min-width: 148px !important;
  flex: 0 0 auto !important;
}

/* Final Video Stage */
#stage-section {
  margin-top: 18px;
  padding: 18px;
  border: 1px solid #232328;
  border-radius: 20px;
  background: #111113;
}
#stage {
  max-width: 420px !important;
  margin: 0 auto !important;
  border: 1px solid #27272c !important;
  border-radius: 16px !important;
  overflow: hidden;
  background: #0b0b0c !important;
}
#stage video { border-radius: 16px; }

/* Optional Engine Settings Drawer */
#engine-settings {
  margin-top: 18px;
  border: 1px solid #1e1e22 !important;
  border-radius: 14px !important;
  background: #111113 !important;
}
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.neutral,
    neutral_hue=gr.themes.colors.neutral,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

_GRADIO6 = "css" in inspect.signature(gr.Blocks.launch).parameters


def _render_header_html(is_ready: bool, detail: str = "") -> str:
    dot_cls = "engine-dot ready" if is_ready else "engine-dot"
    label = detail or ("Engine Ready · GPU Active" if is_ready else "Engine Standby · Auto-Initializes on Preview")
    return (
        '<div id="studio-header">'
        '<div class="brand-wrap">'
        '<span class="brand-name">Wan 2.1 Studio</span>'
        '<span class="brand-sub">35mm Video Synthesis · Tesla T4</span>'
        "</div>"
        f'<div class="engine-pill"><span class="{dot_cls}"></span><span>{html.escape(label)}</span></div>'
        "</div>"
    )


def _hint(length: str) -> str:
    preset = DURATION_PRESETS.get(length, {})
    minutes = preset.get("eta_min", 5.0)
    desc = preset.get("desc", "")
    return f"{desc} · Full render ~{minutes:.0f} min on T4"


def _encode_image_data_uri(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _render_storyboard_html(shots: list, active_idx: int = -1) -> str:
    if not shots:
        return ""
    n = len(shots)
    grid_cls = "sb-grid cols-3" if n > 1 else "sb-grid cols-1"
    unlocked_count = sum(1 for s in shots if s.get("preview_path"))
    header = (
        '<div class="sb-header">'
        '<span class="sb-title">Storyboard Keyframe Preview</span>'
        f'<span class="sb-progress">{unlocked_count} of {n} shots unlocked</span>'
        "</div>"
    )
    cards = []
    for s in shots:
        idx = s["index"]
        is_vert = s["height"] > s["width"]
        aspect_cls = "vertical" if is_vert else "horizontal"
        preview_path = s.get("preview_path")
        if preview_path and os.path.exists(preview_path):
            card_cls = "sb-card unlocked"
            uri = _encode_image_data_uri(preview_path)
            frame_inner = f'<img src="{uri}" alt="{html.escape(s["title"])}" />'
            status_text = f"{s['camera']} · Seed {s['seed']}"
        elif idx == active_idx:
            card_cls = "sb-card active"
            frame_inner = (
                '<div class="sb-placeholder">'
                '<span class="sb-state-tag">Rendering Keyframe</span>'
                f'<span class="sb-shot-cam">{html.escape(s["camera"])}</span>'
                "</div>"
            )
            status_text = "Synthesizing 35mm frame (~6s)..."
        else:
            card_cls = "sb-card"
            frame_inner = (
                '<div class="sb-placeholder">'
                '<span class="sb-state-tag">Locked</span>'
                '<span class="sb-shot-cam">Waiting for previous shot</span>'
                "</div>"
            )
            status_text = s["camera"]

        cards.append(
            f'<div class="{card_cls}">'
            f'<div class="sb-frame {aspect_cls}">{frame_inner}</div>'
            '<div class="sb-meta">'
            f'<span class="sb-shot-name">{html.escape(s["title"])}</span>'
            f'<span class="sb-shot-cam">{html.escape(status_text)}</span>'
            "</div>"
            "</div>"
        )
    return header + f'<div class="{grid_cls}">' + "".join(cards) + "</div>"


def build_demo(pipe=None, audio_pipe=None, output_dir: str = "outputs") -> gr.Blocks:
    state_ref = {
        "pipe": get_cached_pipeline(pipe),
        "audio_pipe": audio_pipe,
    }
    default_length = "Reel 9:16 · Single Shot"

    def _ensure_engine_stream(hf_token_val: str, include_audio: bool):
        """Generator that initializes Wan 2.1 (and optionally AudioLDM 2) with live UI status updates."""
        tok = (hf_token_val or "").strip()
        if tok:
            os.environ["HF_TOKEN"] = tok

        if get_cached_pipeline(state_ref["pipe"]) is None:
            q: queue.Queue = queue.Queue()
            result_holder = {}

            def _worker():
                try:
                    result_holder["pipe"] = load_wan_t4_pipeline(
                        quantization="fast_fp16",
                        hf_token=tok or None,
                        status_cb=lambda m: q.put(m),
                    )
                except Exception as exc:
                    result_holder["error"] = exc
                finally:
                    q.put(None)

            t = threading.Thread(target=_worker, daemon=True)
            t.start()
            while True:
                msg = q.get()
                if msg is None:
                    break
                yield msg
            t.join()
            if "error" in result_holder:
                raise gr.Error(f"Engine initialization failed: {result_holder['error']}")
            state_ref["pipe"] = result_holder["pipe"]
        else:
            state_ref["pipe"] = get_cached_pipeline(state_ref["pipe"])

        if include_audio and state_ref["audio_pipe"] is None:
            yield "Loading AudioLDM 2 soundtrack synthesizer onto CPU RAM..."
            try:
                state_ref["audio_pipe"] = load_audio_pipeline(device="cpu")
            except Exception as e:
                print(f"[Audio Loader] skipped: {e}")

    def run_manual_init(hf_token_val: str, include_audio: bool):
        for msg in _ensure_engine_stream(hf_token_val, include_audio):
            yield _render_header_html(False, "Initializing Engine..."), msg
        yield (
            _render_header_html(True, "Engine Ready · Wan 2.1 FP16 · T4"),
            f"Engine initialized and ready on GPU. {get_gpu_memory_summary()}",
        )

    def run_preview_storyboard(prompt_text: str, length_name: str, hf_token_val: str, include_audio: bool, seed_override: int = -1):
        prompt_text = (prompt_text or "").strip()
        if not prompt_text:
            raise gr.Error("Enter a scene description before generating a preview.")

        # 1. Auto-initialize engine inside Gradio if not loaded yet
        if get_cached_pipeline(state_ref["pipe"]) is None:
            for msg in _ensure_engine_stream(hf_token_val, include_audio):
                yield (
                    _render_header_html(False, "Initializing Engine..."),
                    msg,
                    gr.update(visible=False),
                    "",
                    [],
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                )

        active_pipe = get_cached_pipeline(state_ref["pipe"])
        header_ready = _render_header_html(True, "Engine Ready · Wan 2.1 FP16 · T4")

        # 2. Plan 1-shot or 3-shot storyboard
        yield (
            header_ready,
            "Planning 35mm camera framing and natural film lighting...",
            gr.update(visible=True),
            "",
            [],
            gr.update(interactive=False),
            gr.update(interactive=False),
        )
        shots = plan_storyboard(prompt_text, duration=length_name, seed=seed_override)
        n_shots = len(shots)

        # 3. Unlock each shot preview step-by-step
        for idx, shot in enumerate(shots, start=1):
            yield (
                header_ready,
                f"Rendering keyframe preview for {shot['title']} ({idx} of {n_shots})...",
                gr.update(visible=True),
                _render_storyboard_html(shots, active_idx=idx),
                shots,
                gr.update(interactive=False),
                gr.update(interactive=False),
            )
            render_shot_preview(active_pipe, shot, output_dir=output_dir)
            unlocked_msg = (
                f"Unlocked {shot['title']} ({idx} of {n_shots})."
                if idx < n_shots
                else f"Storyboard complete ({n_shots} of {n_shots} shots unlocked). Review framing or click Render Full Video."
            )
            yield (
                header_ready,
                unlocked_msg,
                gr.update(visible=True),
                _render_storyboard_html(shots, active_idx=-1),
                shots,
                gr.update(interactive=(idx == n_shots)),
                gr.update(interactive=(idx == n_shots)),
            )

    def run_reroll_preview(prompt_text: str, length_name: str, hf_token_val: str, include_audio: bool):
        new_seed = random.randint(0, 2**31 - 100)
        yield from run_preview_storyboard(prompt_text, length_name, hf_token_val, include_audio, seed_override=new_seed)

    def run_full_video(shots_state: list, prompt_text: str, length_name: str, hf_token_val: str, include_audio: bool):
        if not shots_state:
            raise gr.Error("Generate a Storyboard Preview first to lock your camera shots.")

        for msg in _ensure_engine_stream(hf_token_val, include_audio):
            yield (
                _render_header_html(False, "Initializing Engine..."),
                msg,
                gr.update(visible=True),
                None,
            )

        active_pipe = get_cached_pipeline(state_ref["pipe"])
        active_audio = state_ref["audio_pipe"] if include_audio else None
        header_ready = _render_header_html(True, "Engine Ready · Wan 2.1 FP16 · T4")

        q: queue.Queue = queue.Queue()
        result_holder = {}

        def _video_worker():
            try:
                path, seed, secs, has_snd = render_storyboard_video(
                    active_pipe,
                    shots=shots_state,
                    audio_pipe=active_audio,
                    output_dir=output_dir,
                    progress_cb=lambda m: q.put(m),
                )
                result_holder["res"] = (path, seed, secs, has_snd)
            except Exception as exc:
                result_holder["error"] = exc
            finally:
                q.put(None)

        t = threading.Thread(target=_video_worker, daemon=True)
        t.start()
        while True:
            msg = q.get()
            if msg is None:
                break
            yield (
                header_ready,
                msg,
                gr.update(visible=True),
                gr.update(),
            )
        t.join()

        if "error" in result_holder:
            raise gr.Error(f"Video rendering failed: {result_holder['error']}")

        path, seed, secs, has_snd = result_holder["res"]
        n_shots = len(shots_state)
        sound_tag = " · with ambient sound" if has_snd else ""
        summary = f"Mastered 720p HD ({n_shots} {'shot' if n_shots == 1 else 'shots'}) in {secs / 60:.1f} min · Seed {seed}{sound_tag}"
        yield (
            header_ready,
            summary,
            gr.update(visible=True),
            path,
        )

    is_initially_ready = state_ref["pipe"] is not None
    blocks_kwargs = {} if _GRADIO6 else {"theme": THEME, "css": CSS}

    with gr.Blocks(title="Wan 2.1 Studio", **blocks_kwargs) as demo:
        shots_state = gr.State([])

        header_html = gr.HTML(_render_header_html(is_initially_ready))

        with gr.Group(elem_id="composer"):
            prompt = gr.Textbox(
                show_label=False,
                container=False,
                lines=2,
                max_lines=5,
                placeholder="Describe your scene, subject, lighting, and movement...",
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
                preview_btn = gr.Button(
                    "Preview Storyboard",
                    variant="primary",
                    scale=0,
                    elem_id="preview-btn",
                )

        hint = gr.Markdown(_hint(default_length), elem_id="hint")

        with gr.Row(elem_id="chips"):
            for label, text in EXAMPLES.items():
                chip = gr.Button(label, size="sm", variant="secondary", elem_classes="chip")
                chip.click(lambda t=text: t, inputs=None, outputs=prompt, queue=False)

        status = gr.Markdown("", elem_id="status")

        with gr.Column(visible=False, elem_id="storyboard-section") as storyboard_section:
            storyboard_html = gr.HTML("")
            with gr.Row(elem_id="storyboard-actions"):
                reroll_btn = gr.Button(
                    "New Preview Take",
                    variant="secondary",
                    interactive=False,
                    elem_id="reroll-btn",
                )
                render_btn = gr.Button(
                    "Render Full Video",
                    variant="primary",
                    interactive=False,
                    elem_id="render-btn",
                )

        with gr.Column(visible=False, elem_id="stage-section") as stage_section:
            video = gr.Video(
                show_label=False,
                autoplay=True,
                loop=True,
                interactive=False,
                elem_id="stage",
            )

        with gr.Accordion("Engine & Studio Settings", open=False, elem_id="engine-settings"):
            with gr.Row():
                hf_token_box = gr.Textbox(
                    label="Hugging Face Token (Optional — enables Qwen 2.5-72B Cloud Prompt Director & fast downloads)",
                    type="password",
                    value=os.environ.get("HF_TOKEN", ""),
                    placeholder="hf_...",
                    scale=3,
                )
                include_audio_chk = gr.Checkbox(
                    label="Synthesize Ambient Sound (AudioLDM 2)",
                    value=bool(audio_pipe is not None),
                    scale=1,
                )
            init_btn = gr.Button("Initialize Engine in GPU Memory", variant="secondary", size="sm")

        length.change(_hint, inputs=length, outputs=hint, queue=False)

        init_btn.click(
            run_manual_init,
            inputs=[hf_token_box, include_audio_chk],
            outputs=[header_html, status],
            concurrency_limit=1,
        )

        preview_btn.click(
            run_preview_storyboard,
            inputs=[prompt, length, hf_token_box, include_audio_chk],
            outputs=[
                header_html,
                status,
                storyboard_section,
                storyboard_html,
                shots_state,
                reroll_btn,
                render_btn,
            ],
            concurrency_limit=1,
        )

        reroll_btn.click(
            run_reroll_preview,
            inputs=[prompt, length, hf_token_box, include_audio_chk],
            outputs=[
                header_html,
                status,
                storyboard_section,
                storyboard_html,
                shots_state,
                reroll_btn,
                render_btn,
            ],
            concurrency_limit=1,
        )

        render_btn.click(
            run_full_video,
            inputs=[shots_state, prompt, length, hf_token_box, include_audio_chk],
            outputs=[header_html, status, stage_section, video],
            concurrency_limit=1,
        )

    return demo


def launch(demo: gr.Blocks, share: bool = True, **kwargs):
    launch_kwargs = {"theme": THEME, "css": CSS} if _GRADIO6 else {}
    launch_kwargs.update(kwargs)
    return demo.queue(default_concurrency_limit=1, max_size=8).launch(
        share=share, show_error=True, **launch_kwargs
    )


if __name__ == "__main__":
    launch(build_demo(pipe=None), inbrowser=True)
