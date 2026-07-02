"""
SafetyLens AI — Construction Site Safety
--------------------------------------------------
Run with:  streamlit run app.py

Expects a YOLO checkpoint named `ppe.pt` in the same folder as this file.
Class names are read directly from the model (model.names), so this works
with any PPE-style dataset — including ones that include "NO-<item>"
violation classes (e.g. the common Construction Site Safety dataset:
Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Vest,
machinery, vehicle). When violation classes are present, the app also
computes a live Compliance Score.

Optional dependency: streamlit-webrtc + av, for the browser-based Live
Webcam mode. Install with `pip install streamlit-webrtc av`.
"""

import os
import time
import tempfile
import threading
from collections import Counter, defaultdict
from datetime import datetime

import cv2
import torch
import streamlit as st
from PIL import Image
from ultralytics import YOLO

try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False


# =========================================================
# Page config (must run first)
# =========================================================
st.set_page_config(
    page_title="SafetyLens AI",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "ppe.pt"


# =========================================================
# Design system — construction-site signage aesthetic
# =========================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --ppe-bg: #14171C;
    --ppe-surface: #1D2229;
    --ppe-border: #2E343D;
    --ppe-yellow: #F4C430;
    --ppe-orange: #E85D2C;
    --ppe-green: #3FA34D;
    --ppe-red: #C1443B;
    --ppe-text: #ECEAE4;
    --ppe-text-muted: #8891A0;
}

html, body, [data-testid="stAppViewContainer"], .stApp {
    background-color: var(--ppe-bg) !important;
    color: var(--ppe-text);
    font-family: 'IBM Plex Sans', sans-serif;
}
[data-testid="stHeader"] { background-color: transparent; }
[data-testid="stSidebar"] {
    background-color: var(--ppe-surface);
    border-right: 1px solid var(--ppe-border);
}
[data-testid="stSidebar"] * { color: var(--ppe-text); }
p, span, label, li { color: var(--ppe-text); }

/* Hazard stripe — signature element, used sparingly */
.hazard-stripe {
    height: 8px;
    width: 100%;
    background: repeating-linear-gradient(135deg, var(--ppe-yellow) 0 18px, #14171C 18px 36px);
    border-radius: 2px;
    margin-bottom: 1.75rem;
    opacity: 0.92;
}

.ppe-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-size: 0.72rem;
    color: var(--ppe-yellow);
}
.ppe-hero-title {
    font-family: 'Oswald', sans-serif;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.02em;
    font-size: 2.9rem;
    line-height: 1.05;
    margin: 0.35rem 0 0.7rem 0;
    color: var(--ppe-text);
}
.ppe-hero-desc {
    color: var(--ppe-text-muted);
    font-size: 1rem;
    max-width: 660px;
    line-height: 1.6;
}
.ppe-badge-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 1.1rem; }
.ppe-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    border: 1px solid var(--ppe-border);
    background: var(--ppe-surface);
    color: var(--ppe-text-muted);
    padding: 0.3rem 0.6rem;
    border-radius: 3px;
    letter-spacing: 0.03em;
}

.ppe-section-header {
    font-family: 'Oswald', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 1.05rem;
    color: var(--ppe-text);
    border-bottom: 1px solid var(--ppe-border);
    padding-bottom: 0.4rem;
    margin: 1.8rem 0 1rem 0;
}

/* Metrics readout strip */
.ppe-metrics-strip {
    display: flex;
    background: var(--ppe-surface);
    border: 1px solid var(--ppe-border);
    border-radius: 6px;
    overflow: hidden;
    margin: 1.1rem 0;
}
.ppe-metric { flex: 1; padding: 0.9rem 1.2rem; border-right: 1px solid var(--ppe-border); }
.ppe-metric:last-child { border-right: none; }
.ppe-metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--ppe-text-muted);
    margin-bottom: 0.3rem;
}
.ppe-metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--ppe-text);
}

/* Compliance banner */
.ppe-score-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--ppe-surface);
    border: 1px solid var(--ppe-border);
    border-radius: 6px;
    padding: 1rem 1.4rem;
    margin: 1rem 0 1.4rem 0;
}
.ppe-score-value { font-family: 'IBM Plex Mono', monospace; font-size: 2.1rem; font-weight: 700; }

/* Detection cards — inspection-tag styling */
.ppe-card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
    gap: 0.85rem;
    margin: 0.2rem 0 1.6rem 0;
}
.ppe-card {
    background: var(--ppe-surface);
    border: 1px solid var(--ppe-border);
    border-left: 4px solid var(--card-accent, var(--ppe-yellow));
    border-radius: 4px;
    padding: 0.9rem 1rem;
    animation: ppeFadeIn 0.35s ease both;
}
.ppe-card-label {
    font-family: 'Oswald', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    font-size: 0.85rem;
    color: var(--ppe-text);
    margin-bottom: 0.35rem;
}
.ppe-card-count { font-family: 'IBM Plex Mono', monospace; font-size: 1.7rem; font-weight: 600; color: var(--ppe-text); }
.ppe-card-sub { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: var(--ppe-text-muted); margin-top: 0.25rem; }
.ppe-tag-compliant { --card-accent: var(--ppe-green); }
.ppe-tag-violation { --card-accent: var(--ppe-orange); }
.ppe-tag-neutral { --card-accent: var(--ppe-text-muted); }

@keyframes ppeFadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

/* Buttons */
[data-testid="stButton"] button, [data-testid="stDownloadButton"] button {
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    font-size: 0.75rem;
    background: var(--ppe-yellow);
    color: #14171C;
    border: none;
    border-radius: 3px;
    font-weight: 600;
}
[data-testid="stButton"] button:hover, [data-testid="stDownloadButton"] button:hover { background: #ffd968; color: #14171C; }

/* Live indicator */
.ppe-live-dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: var(--ppe-orange); margin-right: 6px;
    animation: ppePulse 1.4s infinite;
}
@keyframes ppePulse {
    0% { box-shadow: 0 0 0 0 rgba(232,93,44,0.55); }
    70% { box-shadow: 0 0 0 8px rgba(232,93,44,0); }
    100% { box-shadow: 0 0 0 0 rgba(232,93,44,0); }
}

.ppe-footer {
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid var(--ppe-border);
    color: var(--ppe-text-muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    display: flex;
    justify-content: space-between;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================================================
# Classification of model classes → compliant / violation / neutral
# =========================================================
NEUTRAL_CLASSES = {"person", "vehicle", "machinery", "safety cone"}


def classify_label(name: str) -> str:
    lname = name.strip().lower()
    if lname.startswith("no-") or lname.startswith("no_") or lname.startswith("no "):
        return "violation"
    if lname in NEUTRAL_CLASSES:
        return "neutral"
    return "compliant"


CARD_CSS_MAP = {"compliant": "ppe-tag-compliant", "violation": "ppe-tag-violation", "neutral": "ppe-tag-neutral"}


# =========================================================
# Model loading
# =========================================================
@st.cache_resource
def load_model(path: str):
    return YOLO(path)


try:
    with st.spinner("Loading detection model..."):
        model = load_model(MODEL_PATH)
except Exception as e:
    st.markdown(
        f"""
        <div class="ppe-card ppe-tag-violation" style="max-width:640px;">
            <div class="ppe-card-label">Model Not Found</div>
            <div style="margin-top:0.5rem; font-family:'IBM Plex Sans',sans-serif; font-size:0.85rem; color:var(--ppe-text); line-height:1.5;">
                Couldn't load <code>{MODEL_PATH}</code>. Place your trained weights file
                in the same folder as <code>app.py</code> and restart the app.
                <br><br>
                <span style="color:var(--ppe-text-muted); font-family:'IBM Plex Mono',monospace; font-size:0.72rem;">{e}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

MODEL_NAMES = model.names  # {int: str}, read from the checkpoint — no hardcoded class list
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HAS_VIOLATION_CLASSES = any(classify_label(n) == "violation" for n in MODEL_NAMES.values())


# =========================================================
# Reusable render helpers
# =========================================================
def render_hero():
    st.markdown('<div class="hazard-stripe"></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="ppe-eyebrow">Computer Vision · Site Safety Compliance</div>
        <div class="ppe-hero-title">SafetyLens AI</div>
        <div class="ppe-hero-desc">
            Automated identification of personal protective equipment on construction
            sites, powered by a fine-tuned YOLO object detector. Run detection on an
            image, a video file, or a live webcam feed to surface required gear and
            flag safety violations as they happen.
        </div>
        <div class="ppe-badge-row">
            <span class="ppe-badge">MODEL: Yolov8s</span>
            <span class="ppe-badge">{len(MODEL_NAMES)} CLASSES</span>
            <span class="ppe-badge">DEVICE: {DEVICE.upper()}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics_strip(inference_ms: float, fps: float, total_detections: int, num_classes: int):
    st.markdown(
        f"""
        <div class="ppe-metrics-strip">
            <div class="ppe-metric"><div class="ppe-metric-label">Inference Time</div><div class="ppe-metric-value">{inference_ms:.1f} ms</div></div>
            <div class="ppe-metric"><div class="ppe-metric-label">Throughput</div><div class="ppe-metric-value">{fps:.1f} FPS</div></div>
            <div class="ppe-metric"><div class="ppe-metric-label">Detections</div><div class="ppe-metric-value">{total_detections}</div></div>
            <div class="ppe-metric"><div class="ppe-metric-label">Classes Found</div><div class="ppe-metric-value">{num_classes}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_compliance_banner(counter: Counter):
    compliant = sum(v for k, v in counter.items() if classify_label(k) == "compliant")
    violations = sum(v for k, v in counter.items() if classify_label(k) == "violation")
    total = compliant + violations

    if total == 0:
        score_display, color = "—", "var(--ppe-text-muted)"
        note = "No compliance-relevant classes detected yet."
    else:
        score = 100 * compliant / total
        score_display = f"{score:.0f}%"
        color = "var(--ppe-green)" if score >= 90 else "var(--ppe-orange)" if score >= 70 else "var(--ppe-red)"
        note = f"{compliant} compliant item(s) vs. {violations} flagged violation(s)."

    st.markdown(
        f"""
        <div class="ppe-score-banner">
            <div>
                <div class="ppe-metric-label">Compliance Score</div>
                <div style="color:var(--ppe-text-muted); font-family:'IBM Plex Sans',sans-serif; font-size:0.82rem; margin-top:0.25rem;">{note}</div>
            </div>
            <div class="ppe-score-value" style="color:{color};">{score_display}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_cards(counter: Counter, avg_conf: dict):
    cards = []
    for label, count in sorted(counter.items(), key=lambda kv: -kv[1]):
        ctype = classify_label(label)
        conf = avg_conf.get(label)
        conf_str = f"{conf * 100:.1f}% avg. confidence" if conf is not None else ""
        cards.append(
            f'<div class="ppe-card {CARD_CSS_MAP[ctype]}">'
            f'<div class="ppe-card-label">{label}</div>'
            f'<div class="ppe-card-count">{count}</div>'
            f'<div class="ppe-card-sub">{conf_str}</div>'
            f"</div>"
        )
    st.markdown(f'<div class="ppe-card-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def tally_boxes(results):
    """Count detections per class label and accumulate confidence sums."""
    counter, conf_sums = Counter(), defaultdict(float)
    for box in results[0].boxes:
        label = MODEL_NAMES[int(box.cls[0])]
        counter[label] += 1
        conf_sums[label] += float(box.conf[0])
    return counter, conf_sums


# =========================================================
# Sidebar — control panel
# =========================================================
with st.sidebar:
    st.markdown('<div class="ppe-eyebrow" style="margin-bottom:0.3rem;">Control Panel</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family:\'Oswald\',sans-serif; text-transform:uppercase; '
        'font-size:1.1rem; letter-spacing:0.03em; margin-bottom:1rem;">Inspection Settings</div>',
        unsafe_allow_html=True,
    )

    mode = st.radio("Detection Mode", ("Image", "Video", "Webcam"))

    st.markdown("---")
    confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05, help="Minimum score for a detection to be kept.")
    iou = st.slider("IoU Threshold", 0.1, 1.0, 0.45, 0.05, help="Overlap threshold used for non-max suppression.")

    frame_skip, show_live_preview = 1, True
    if mode == "Video":
        st.markdown("---")
        frame_skip = st.number_input(
            "Run inference every Nth frame", min_value=1, max_value=10, value=2,
            help="Higher values process faster but update detections less often. Frames in between reuse the last result.",
        )
        show_live_preview = st.checkbox("Show live preview while processing", value=True)

    st.markdown("---")
    with st.expander("Model Info"):
        badges = "".join(f'<span class="ppe-badge">{n}</span>' for n in MODEL_NAMES.values())
        st.markdown(f'<div class="ppe-badge-row">{badges}</div>', unsafe_allow_html=True)
        st.caption(f"{len(MODEL_NAMES)} classes · running on {DEVICE.upper()}")

    with st.expander("How it works"):
        st.markdown(
            "Each frame is passed through a YOLO detector fine-tuned on annotated "
            "construction-site imagery. Boxes below the confidence threshold are "
            "discarded, and overlapping boxes for the same object are merged using "
            "the IoU threshold. If the model was trained with explicit **NO-*** "
            "violation classes, detected items are also scored as compliant or "
            "flagged, and rolled up into a Compliance Score."
        )
    st.markdown("---")
    st.markdown("SafetyLens AI")
    st.markdown("Built by Zeyan Nameer")

# =========================================================
# Main content
# =========================================================
render_hero()

if mode == "Image":
    st.markdown('<div class="ppe-section-header">Image Detection</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a site photo", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        t0 = time.time()
        results = model.predict(image, conf=confidence, iou=iou, verbose=False)
        inference_ms = (time.time() - t0) * 1000
        fps = 1000 / inference_ms if inference_ms > 0 else 0.0

        annotated_bgr = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        counter, conf_sums = tally_boxes(results)
        avg_conf = {k: conf_sums[k] / counter[k] for k in counter}

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="ppe-metric-label" style="margin-bottom:0.5rem;">ORIGINAL</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
        with col2:
            st.markdown('<div class="ppe-metric-label" style="margin-bottom:0.5rem;">DETECTION RESULT</div>', unsafe_allow_html=True)
            st.image(annotated_rgb, use_container_width=True)

        render_metrics_strip(inference_ms, fps, sum(counter.values()), len(counter))
        if HAS_VIOLATION_CLASSES:
            render_compliance_banner(counter)

        st.markdown('<div class="ppe-section-header">Detection Summary</div>', unsafe_allow_html=True)
        if counter:
            render_cards(counter, avg_conf)
        else:
            st.info("No PPE items detected at the current confidence threshold.")

        png_bytes = cv2.imencode(".png", annotated_bgr)[1].tobytes()
        st.download_button(
            "Download Annotated Image",
            data=png_bytes,
            file_name=f"ppe_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png",
        )
    else:
        st.info("Upload a JPG or PNG image of a construction site to run detection.")


elif mode == "Video":
    st.markdown('<div class="ppe-section-header">Video Detection</div>', unsafe_allow_html=True)
    uploaded_video = st.file_uploader("Upload a site video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        run_video = st.button("Run Detection on Video")

        if run_video:
            in_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            in_tmp.write(uploaded_video.read())
            in_tmp.close()

            cap = cv2.VideoCapture(in_tmp.name)
            fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (width, height))

            progress = st.progress(0, text="Processing video...")
            preview = st.empty() if show_live_preview else None

            counter, conf_sums = Counter(), defaultdict(float)
            inference_times = []
            last_annotated, frame_idx = None, 0

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                if frame_idx % frame_skip == 0 or last_annotated is None:
                    t0 = time.time()
                    results = model.predict(frame, conf=confidence, iou=iou, verbose=False)
                    inference_times.append((time.time() - t0) * 1000)
                    last_annotated = results[0].plot()
                    frame_counter, frame_conf_sums = tally_boxes(results)
                    counter.update(frame_counter)
                    for k, v in frame_conf_sums.items():
                        conf_sums[k] += v

                writer.write(last_annotated)
                if preview is not None and frame_idx % 3 == 0:
                    preview.image(cv2.cvtColor(last_annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

                frame_idx += 1
                if total_frames:
                    progress.progress(min(frame_idx / total_frames, 1.0), text=f"Processing frame {frame_idx}/{total_frames}")

            cap.release()
            writer.release()
            progress.empty()
            if preview is not None:
                preview.empty()

            avg_ms = sum(inference_times) / len(inference_times) if inference_times else 0
            avg_fps = 1000 / avg_ms if avg_ms > 0 else 0
            render_metrics_strip(avg_ms, avg_fps, sum(counter.values()), len(counter))
            if HAS_VIOLATION_CLASSES:
                render_compliance_banner(counter)

            st.markdown('<div class="ppe-section-header">Detection Summary</div>', unsafe_allow_html=True)
            if counter:
                avg_conf = {k: conf_sums[k] / counter[k] for k in counter}
                render_cards(counter, avg_conf)
            else:
                st.info("No PPE items detected in this video at the current threshold.")

            with open(out_path, "rb") as f:
                video_bytes = f.read()
            st.download_button(
                "Download Annotated Video",
                data=video_bytes,
                file_name=f"ppe_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                mime="video/mp4",
            )

            os.unlink(in_tmp.name)
            os.unlink(out_path)
    else:
        st.info("Upload an MP4, AVI, or MOV file to run detection on video.")


elif mode == "Webcam":
    st.markdown('<div class="ppe-section-header">Live Webcam Detection</div>', unsafe_allow_html=True)

    if not WEBRTC_AVAILABLE:
        st.warning(
            "Live webcam mode needs the `streamlit-webrtc` and `av` packages.\n\n"
            "Install them with:\n\n`pip install streamlit-webrtc av`\n\nthen restart the app."
        )
    else:
        RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

        class PPEVideoProcessor(VideoProcessorBase):
            def __init__(self):
                self.conf = confidence
                self.iou = iou
                self.lock = threading.Lock()
                self.stats = {"inference_ms": 0.0, "fps": 0.0, "counter": Counter(), "conf_sums": defaultdict(float)}

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                t0 = time.time()
                results = model.predict(img, conf=self.conf, iou=self.iou, verbose=False)
                inference_ms = (time.time() - t0) * 1000
                annotated = results[0].plot()
                counter, conf_sums = tally_boxes(results)

                with self.lock:
                    self.stats = {
                        "inference_ms": inference_ms,
                        "fps": 1000 / inference_ms if inference_ms > 0 else 0,
                        "counter": counter,
                        "conf_sums": conf_sums,
                    }

                return av.VideoFrame.from_ndarray(annotated, format="bgr24")

        st.caption("Thresholds are applied when the stream starts — stop and restart the stream to apply new slider values.")

        webrtc_ctx = webrtc_streamer(
            key="ppe-live-detection",
            video_processor_factory=PPEVideoProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        status_placeholder = st.empty()
        metrics_placeholder = st.empty()
        cards_placeholder = st.empty()

        if webrtc_ctx.state.playing:
            status_placeholder.markdown(
                '<span class="ppe-live-dot"></span><span class="ppe-eyebrow">LIVE</span>', unsafe_allow_html=True
            )
        else:
            status_placeholder.markdown(
                '<span class="ppe-eyebrow" style="color:var(--ppe-text-muted);">IDLE — click Start above</span>',
                unsafe_allow_html=True,
            )

        while webrtc_ctx.state.playing:
            if webrtc_ctx.video_processor:
                with webrtc_ctx.video_processor.lock:
                    stats = dict(webrtc_ctx.video_processor.stats)

                counter = stats.get("counter", Counter())
                conf_sums = stats.get("conf_sums", {})
                avg_conf = {k: conf_sums[k] / counter[k] for k in counter} if counter else {}

                with metrics_placeholder.container():
                    render_metrics_strip(stats.get("inference_ms", 0), stats.get("fps", 0), sum(counter.values()), len(counter))
                    if HAS_VIOLATION_CLASSES:
                        render_compliance_banner(counter)

                with cards_placeholder.container():
                    if counter:
                        render_cards(counter, avg_conf)
                    else:
                        st.caption("No PPE items detected in the current frame.")

            time.sleep(0.7)


# =========================================================
# Footer
# =========================================================
st.markdown(
    f"""
    <div class="ppe-footer">
        <span>SafetyLens AI · YOLO + Streamlit</span>
        <span>Session started {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
    </div>
    """,
    unsafe_allow_html=True,
)
