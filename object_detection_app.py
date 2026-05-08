"""
Multi-Class Object Detection System (YOLOv8 + Streamlit)
Supports: Image Upload + Live Webcam modes + Scene Understanding
Uses streamlit-webrtc for browser-compatible webcam access.
"""

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import json
import time
import threading
import av
from collections import Counter
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from scene_classifier import classify_scene, get_scene_overlay_text

# ── Page Config ──
st.set_page_config(page_title="🔍 Object Detection Studio", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")

# ── CSS ──
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ═══════════════════════════════════════════════════════════
   BASE & TYPOGRAPHY
   ═══════════════════════════════════════════════════════════ */
html, body, .stApp { font-family: 'Inter', sans-serif; }
.stApp {
  background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
  color: #e2e8f0;
}

/* Force all top-level text light */
.stApp p, .stApp span, .stApp label, .stApp li,
.stApp div[data-testid="stMarkdownContainer"] p,
.stApp div[data-testid="stMarkdownContainer"] li {
  color: #e2e8f0 !important;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
  color: #f1f5f9 !important;
}
.stApp small, .stApp .stCaption {
  color: rgba(255,255,255,.55) !important;
}

/* ═══════════════════════════════════════════════════════════
   HERO HEADER
   ═══════════════════════════════════════════════════════════ */
.hero-header {
  text-align: center; padding: 2rem 1rem 1rem;
  background: linear-gradient(135deg, rgba(99,102,241,.18), rgba(168,85,247,.18));
  border-radius: 20px; border: 1px solid rgba(255,255,255,.10);
  margin-bottom: 1.5rem;
  box-shadow: 0 4px 30px rgba(99,102,241,.08);
}
.hero-header h1 {
  font-size: 2.4rem; font-weight: 800;
  background: linear-gradient(135deg, #818cf8, #a78bfa, #c084fc);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  margin-bottom: .3rem;
}
.hero-header p { color: rgba(255,255,255,.65) !important; font-size: 1.05rem; }

/* ═══════════════════════════════════════════════════════════
   GLASS CARD
   ═══════════════════════════════════════════════════════════ */
.glass-card {
  background: rgba(255,255,255,.05); backdrop-filter: blur(16px);
  border-radius: 16px; border: 1px solid rgba(255,255,255,.10);
  padding: 1.5rem; margin-bottom: 1rem;
  box-shadow: 0 2px 20px rgba(0,0,0,.15);
}

/* ═══════════════════════════════════════════════════════════
   METRIC CARDS
   ═══════════════════════════════════════════════════════════ */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap; }
.metric-card {
  flex: 1; min-width: 140px; background: rgba(255,255,255,.06);
  border-radius: 14px; padding: 1.1rem; text-align: center;
  border: 1px solid rgba(255,255,255,.10);
  transition: transform .2s ease, box-shadow .2s ease;
}
.metric-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 8px 30px rgba(99,102,241,.25);
}
.metric-card .label { font-size: .8rem; color: rgba(255,255,255,.6) !important; margin-bottom: .3rem; font-weight: 500; }
.metric-card .value { font-size: 1.6rem; font-weight: 700; }
.metric-card .value.purple { color: #a78bfa; }
.metric-card .value.green  { color: #34d399; }
.metric-card .value.blue   { color: #60a5fa; }
.metric-card .value.amber  { color: #fbbf24; }

/* ═══════════════════════════════════════════════════════════
   DETECTION LIST
   ═══════════════════════════════════════════════════════════ */
.det-item {
  display: flex; justify-content: space-between; align-items: center;
  padding: .7rem 1rem; margin-bottom: .45rem; border-radius: 10px;
  background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.08);
  color: #f1f5f9 !important; font-size: .88rem; font-weight: 400;
  transition: background .15s ease;
}
.det-item:hover { background: rgba(255,255,255,.08); }
.det-badge {
  display: inline-block; padding: .2rem .6rem; border-radius: 8px;
  font-size: .75rem; font-weight: 600;
}
.badge-person { background: rgba(52,211,153,.2); color: #34d399; }
.badge-object { background: rgba(96,165,250,.2); color: #60a5fa; }

/* ═══════════════════════════════════════════════════════════
   SCENE BOXES & CARDS
   ═══════════════════════════════════════════════════════════ */
.scene-box {
  background: linear-gradient(135deg, rgba(99,102,241,.14), rgba(168,85,247,.14));
  border-radius: 14px; padding: 1.2rem 1.5rem;
  border: 1px solid rgba(168,85,247,.3);
  color: #f1f5f9 !important; font-size: .95rem; line-height: 1.6; margin-bottom: 1rem;
}
.scene-card {
  background: linear-gradient(135deg, rgba(16,185,129,.12), rgba(59,130,246,.12));
  border-radius: 14px; padding: 1.2rem 1.5rem;
  border: 1px solid rgba(16,185,129,.3); margin-bottom: 1rem;
}
.scene-card .scene-label { font-size: 1.3rem; font-weight: 700; color: #34d399; margin-bottom: .4rem; }
.scene-card .scene-desc  { color: #e2e8f0 !important; font-size: .92rem; line-height: 1.5; }
.scene-card .scene-tags  { margin-top: .6rem; display: flex; gap: .5rem; flex-wrap: wrap; }
.scene-tag {
  background: rgba(255,255,255,.1); color: #c4b5fd !important;
  padding: .25rem .65rem; border-radius: 8px; font-size: .75rem; font-weight: 500;
}
.scene-scores { display: flex; gap: 1rem; margin-top: .6rem; flex-wrap: wrap; }
.score-pill   { padding: .3rem .7rem; border-radius: 8px; font-size: .75rem; font-weight: 600; }
.score-nature { background: rgba(52,211,153,.18); color: #34d399; }
.score-urban  { background: rgba(96,165,250,.18); color: #60a5fa; }
.score-indoor { background: rgba(251,191,36,.18); color: #fbbf24; }

/* ═══════════════════════════════════════════════════════════
   SIDEBAR — Background & Global Text
   ═══════════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #1e1b4b, #1a1a3e) !important;
  border-right: 1px solid rgba(255,255,255,.08);
}
section[data-testid="stSidebar"] * { color: #f1f5f9 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #c4b5fd !important; font-weight: 700; }
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div { color: #f1f5f9 !important; }
section[data-testid="stSidebar"] small { color: rgba(255,255,255,.55) !important; }
section[data-testid="stSidebar"] hr {
  border-color: rgba(255,255,255,.12) !important;
  margin: .6rem 0;
}

/* ═══════════════════════════════════════════════════════════
   SELECTBOX / DROPDOWN (YOLO Model & others)
   ═══════════════════════════════════════════════════════════ */
/* --- Container --- */
div[data-baseweb="select"] > div {
  background: rgba(255,255,255,.08) !important;
  border: 1px solid rgba(167,139,250,.35) !important;
  border-radius: 10px !important;
  color: #f1f5f9 !important;
  transition: border-color .2s ease, box-shadow .2s ease;
}
div[data-baseweb="select"] > div:hover {
  border-color: rgba(167,139,250,.6) !important;
  box-shadow: 0 0 12px rgba(167,139,250,.15);
}
div[data-baseweb="select"] > div:focus-within {
  border-color: #a78bfa !important;
  box-shadow: 0 0 0 3px rgba(167,139,250,.2) !important;
}
/* --- Selected value text --- */
div[data-baseweb="select"] span,
div[data-baseweb="select"] div[data-testid="stMarkdownContainer"] {
  color: #ffffff !important; font-weight: 500;
}
/* --- Dropdown arrow --- */
div[data-baseweb="select"] svg { fill: #c4b5fd !important; }
/* --- Dropdown menu --- */
div[data-baseweb="popover"] ul,
div[data-baseweb="menu"] {
  background: #1e1b4b !important; border: 1px solid rgba(167,139,250,.3) !important;
  border-radius: 10px !important;
}
div[data-baseweb="popover"] li,
div[data-baseweb="menu"] li {
  color: #e2e8f0 !important; font-weight: 400;
}
div[data-baseweb="popover"] li:hover,
div[data-baseweb="menu"] li:hover {
  background: rgba(167,139,250,.18) !important; color: #ffffff !important;
}
div[data-baseweb="popover"] li[aria-selected="true"],
div[data-baseweb="menu"] li[aria-selected="true"] {
  background: rgba(167,139,250,.25) !important; color: #ffffff !important; font-weight: 600;
}

/* ═══════════════════════════════════════════════════════════
   FILE UPLOADER — "Browse Files" button & dropzone
   ═══════════════════════════════════════════════════════════ */
/* Force dark on the uploader parent containers */
.stFileUploader, .stFileUploader > div,
.stFileUploader section,
.stFileUploader > section {
  background: transparent !important;
}
div[data-testid="stFileUploaderDropzone"],
.stFileUploader div[data-testid="stFileUploaderDropzone"],
section[data-testid="stFileUploaderDropzone"] {
  background: rgba(30,27,75,.85) !important;
  border: 2px dashed rgba(167,139,250,.4) !important;
  border-radius: 14px !important;
  padding: 1.8rem 1rem !important;
  transition: border-color .25s ease, background .25s ease, box-shadow .25s ease;
}
div[data-testid="stFileUploaderDropzone"]:hover {
  border-color: rgba(167,139,250,.65) !important;
  background: rgba(30,27,75,.92) !important;
  box-shadow: 0 0 20px rgba(167,139,250,.12);
}
/* Drag-and-drop text — force bright */
div[data-testid="stFileUploaderDropzone"] span,
div[data-testid="stFileUploaderDropzone"] p {
  color: rgba(255,255,255,.75) !important; font-weight: 500;
}
div[data-testid="stFileUploaderDropzone"] small {
  color: rgba(255,255,255,.5) !important;
}
/* Upload cloud icon */
div[data-testid="stFileUploaderDropzone"] svg {
  fill: #a78bfa !important; opacity: .85;
}
/* Browse Files button — force purple gradient always */
div[data-testid="stFileUploaderDropzone"] button,
.stFileUploader button,
div[data-testid="stFileUploaderDropzone"] [data-testid="baseButton-secondary"],
div[data-testid="stFileUploaderDropzone"] button[kind="secondary"],
div[data-testid="stFileUploaderDropzone"] button[data-testid="baseButton-secondary"] {
  background: linear-gradient(135deg, #6366f1, #7c3aed) !important;
  color: #ffffff !important; font-weight: 600 !important; font-size: .88rem !important;
  border: none !important; border-radius: 10px !important;
  padding: .55rem 1.5rem !important;
  box-shadow: 0 4px 15px rgba(99,102,241,.35);
  transition: transform .15s ease, box-shadow .15s ease, background .2s ease;
  letter-spacing: .02em;
}
div[data-testid="stFileUploaderDropzone"] button:hover,
.stFileUploader button:hover {
  background: linear-gradient(135deg, #7c3aed, #8b5cf6) !important;
  transform: translateY(-1px);
  box-shadow: 0 6px 22px rgba(124,58,237,.4);
}
div[data-testid="stFileUploaderDropzone"] button:active {
  transform: translateY(0); box-shadow: 0 2px 8px rgba(99,102,241,.3);
}

/* ═══════════════════════════════════════════════════════════
   RADIO BUTTONS
   ═══════════════════════════════════════════════════════════ */
div[role="radiogroup"] label {
  color: #e2e8f0 !important; font-weight: 400;
  transition: color .15s ease;
}
div[role="radiogroup"] label:hover { color: #ffffff !important; }
div[role="radiogroup"] label[data-checked="true"],
div[role="radiogroup"] label[aria-checked="true"] {
  color: #ffffff !important; font-weight: 600;
}
/* Radio dot */
div[role="radiogroup"] div[data-testid="stMarkdownContainer"] p { color: #f1f5f9 !important; }

/* ═══════════════════════════════════════════════════════════
   CHECKBOXES
   ═══════════════════════════════════════════════════════════ */
.stCheckbox span { color: #f1f5f9 !important; font-weight: 400; }
.stCheckbox label:hover span { color: #ffffff !important; }

/* ═══════════════════════════════════════════════════════════
   SLIDERS
   ═══════════════════════════════════════════════════════════ */
/* Track background */
div[data-baseweb="slider"] div[role="slider"] {
  background: #a78bfa !important;
  border: 2px solid #c4b5fd !important;
  box-shadow: 0 0 8px rgba(167,139,250,.4);
}
div[data-testid="stSlider"] label { color: #f1f5f9 !important; font-weight: 500; }
div[data-testid="stSlider"] div[data-testid="stTickBarMin"],
div[data-testid="stSlider"] div[data-testid="stTickBarMax"] {
  color: rgba(255,255,255,.5) !important;
}
/* Slider current value */
div[data-testid="stSlider"] div[data-testid="stThumbValue"] {
  color: #c4b5fd !important; font-weight: 600;
}

/* ═══════════════════════════════════════════════════════════
   BUTTONS (general Streamlit buttons)
   ═══════════════════════════════════════════════════════════ */
.stButton button, button[data-testid="baseButton-primary"] {
  background: linear-gradient(135deg, #6366f1, #7c3aed) !important;
  color: #ffffff !important; font-weight: 600 !important;
  border: none !important; border-radius: 10px !important;
  padding: .5rem 1.2rem !important;
  box-shadow: 0 4px 15px rgba(99,102,241,.25);
  transition: transform .15s ease, box-shadow .2s ease;
}
.stButton button:hover, button[data-testid="baseButton-primary"]:hover {
  background: linear-gradient(135deg, #7c3aed, #8b5cf6) !important;
  transform: translateY(-1px);
  box-shadow: 0 6px 22px rgba(124,58,237,.35);
}
.stButton button:active { transform: translateY(0); }
/* Secondary/stop buttons */
button[data-testid="baseButton-secondary"] {
  background: rgba(255,255,255,.08) !important;
  color: #e2e8f0 !important; font-weight: 500 !important;
  border: 1px solid rgba(255,255,255,.15) !important; border-radius: 10px !important;
  transition: background .15s ease;
}
button[data-testid="baseButton-secondary"]:hover {
  background: rgba(255,255,255,.12) !important; color: #ffffff !important;
}

/* ═══════════════════════════════════════════════════════════
   EXPANDER
   ═══════════════════════════════════════════════════════════ */
details summary { color: #e2e8f0 !important; }
details summary span { color: #e2e8f0 !important; font-weight: 500; }
details summary:hover span { color: #ffffff !important; }
details summary svg { fill: #a78bfa !important; }

/* ═══════════════════════════════════════════════════════════
   JSON BLOCK
   ═══════════════════════════════════════════════════════════ */
.json-block {
  background: rgba(0,0,0,.4); border-radius: 12px; padding: 1rem;
  font-family: 'Fira Code', 'Cascadia Code', monospace; font-size: .8rem;
  color: #c4b5fd; overflow-x: auto;
  border: 1px solid rgba(255,255,255,.08); max-height: 400px; overflow-y: auto;
}

/* ═══════════════════════════════════════════════════════════
   WEBCAM STATUS BADGES
   ═══════════════════════════════════════════════════════════ */
.webcam-status {
  padding: .8rem 1.2rem; border-radius: 12px;
  text-align: center; font-weight: 600; margin-bottom: 1rem;
}
.webcam-live {
  background: rgba(52,211,153,.15); border: 1px solid rgba(52,211,153,.35);
  color: #34d399;
}
.webcam-off {
  background: rgba(248,113,113,.15); border: 1px solid rgba(248,113,113,.35);
  color: #f87171;
}

/* ═══════════════════════════════════════════════════════════
   TEXT INPUTS (if any)
   ═══════════════════════════════════════════════════════════ */
.stTextInput input, .stNumberInput input {
  background: rgba(255,255,255,.06) !important;
  color: #f1f5f9 !important; border: 1px solid rgba(167,139,250,.3) !important;
  border-radius: 10px !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
  border-color: #a78bfa !important;
  box-shadow: 0 0 0 3px rgba(167,139,250,.15) !important;
}

/* ═══════════════════════════════════════════════════════════
   SCROLLBAR STYLING
   ═══════════════════════════════════════════════════════════ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,.03); }
::-webkit-scrollbar-thumb {
  background: rgba(167,139,250,.3); border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(167,139,250,.5); }

/* ═══════════════════════════════════════════════════════════
   SPINNERS & TOASTS
   ═══════════════════════════════════════════════════════════ */
.stSpinner > div { color: #c4b5fd !important; }
div[data-testid="stNotification"] { background: #1e1b4b !important; color: #e2e8f0 !important; }

/* ═══════════════════════════════════════════════════════════
   TOOLTIP / HELP ICONS
   ═══════════════════════════════════════════════════════════ */
.stTooltipIcon svg { fill: rgba(255,255,255,.4) !important; }
.stTooltipIcon:hover svg { fill: rgba(255,255,255,.7) !important; }

</style>""", unsafe_allow_html=True)

# ── Model Loading ──
@st.cache_resource(show_spinner=False)
def load_model(model_size="yolov8n"):
    return YOLO(f"{model_size}.pt")

COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

# ── Shared Detection Functions ──
def run_detection(model, image_bgr, confidence):
    start = time.perf_counter()
    results = model.predict(image_bgr, conf=confidence, verbose=False)
    elapsed_ms = (time.perf_counter() - start) * 1000
    detections = []
    if results and len(results) > 0:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            label = COCO_NAMES[cls_id] if cls_id < len(COCO_NAMES) else f"class_{cls_id}"
            detections.append({"id":i,"class_id":cls_id,"label":label,"confidence":round(conf,4),
                "bbox":{"x1":int(x1),"y1":int(y1),"x2":int(x2),"y2":int(y2)},"is_person":label=="person"})
    return detections, elapsed_ms

def filter_detections(detections, mode):
    if mode == "People Only": return [d for d in detections if d["is_person"]]
    elif mode == "Objects Only": return [d for d in detections if not d["is_person"]]
    return detections

def annotate_image(image_bgr, detections, scene_label=None):
    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]
    fs = max(0.45, min(w,h)/1200)
    th = max(1, int(min(w,h)/500))
    for det in detections:
        b = det["bbox"]; x1,y1,x2,y2 = b["x1"],b["y1"],b["x2"],b["y2"]
        color = (52,211,153) if det["is_person"] else (250,165,96)
        cv2.rectangle(annotated,(x1,y1),(x2,y2),color,th+1)
        txt = f"{det['label']} {det['confidence']:.0%}"
        (tw,th2),_ = cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,fs,th)
        cv2.rectangle(annotated,(x1,y1-th2-10),(x1+tw+6,y1),color,-1)
        cv2.putText(annotated,txt,(x1+3,y1-5),cv2.FONT_HERSHEY_SIMPLEX,fs,(0,0,0),th,cv2.LINE_AA)
    # Overlay scene label at top
    if scene_label:
        overlay_fs = max(0.55, min(w,h)/800)
        overlay_th = max(1, int(min(w,h)/400))
        (tw2,th3),_ = cv2.getTextSize(scene_label,cv2.FONT_HERSHEY_SIMPLEX,overlay_fs,overlay_th)
        cv2.rectangle(annotated,(0,0),(tw2+16,th3+18),(30,30,30),-1)
        cv2.putText(annotated,scene_label,(8,th3+10),cv2.FONT_HERSHEY_SIMPLEX,overlay_fs,(52,211,153),overlay_th,cv2.LINE_AA)
    return annotated

def generate_scene_summary(detections):
    """Basic object summary (kept for backward compat)."""
    if not detections: return "No objects detected."
    counts = Counter(d["label"] for d in detections)
    pc = counts.pop("person",0)
    obj_parts = [f"{c} {o}{'s' if c>1 else ''}" for o,c in counts.most_common()]
    if pc and obj_parts: s = f"Detected {pc} {'person' if pc==1 else 'people'} along with {', '.join(obj_parts)}."
    elif pc: s = f"Detected {pc} {'person' if pc==1 else 'people'} in the scene."
    else: s = f"Detected {', '.join(obj_parts)}."
    top = max(detections, key=lambda d: d["confidence"])
    return s + f" Most confident: **{top['label']}** ({top['confidence']:.0%})."

def render_scene_card(scene_result):
    """Render the scene classification card with Places365 predictions."""
    sr = scene_result
    tags_html = "".join(f'<span class="scene-tag">{t}</span>' for t in sr["environment_tags"])
    scores = sr["scene_scores"]
    scores_html = (
        f'<span class="score-pill score-nature">Nature: {scores["nature"]}</span>'
        f'<span class="score-pill score-urban">Urban: {scores["urban"]}</span>'
        f'<span class="score-pill score-indoor">Indoor: {scores["indoor"]}</span>'
    )
    # Places365 top predictions
    preds_html = ""
    if sr.get("places365_top5"):
        preds_items = "".join(
            f'<div style="display:flex;align-items:center;gap:.5rem;margin:.2rem 0;">'
            f'<span style="min-width:140px;color:#cbd5e1;font-size:.8rem;">{cat.replace("_"," ").title()}</span>'
            f'<div style="flex:1;background:rgba(255,255,255,.06);border-radius:4px;height:8px;">'
            f'<div style="width:{prob*100:.0f}%;background:linear-gradient(90deg,#818cf8,#a78bfa);'
            f'border-radius:4px;height:100%;"></div></div>'
            f'<span style="color:#a5b4fc;font-size:.75rem;min-width:40px;">{prob:.0%}</span></div>'
            for cat, prob in sr["places365_top5"]
        )
        preds_html = f'<div style="margin-top:.8rem;"><div style="color:rgba(255,255,255,.4);font-size:.72rem;margin-bottom:.3rem;">PLACES365 PREDICTIONS</div>{preds_items}</div>'

    st.markdown(
        f'<div class="scene-card">'
        f'  <div class="scene-label">{sr["scene_label"]}</div>'
        f'  <div class="scene-desc">{sr["description"]}</div>'
        f'  <div class="scene-tags">{tags_html}</div>'
        f'  <div class="scene-scores">{scores_html}</div>'
        f'  {preds_html}'
        f'</div>', unsafe_allow_html=True)

def build_structured_output(detections, elapsed_ms, shape):
    h,w = shape[:2]
    return {"image_info":{"width":w,"height":h},"inference_time_ms":round(elapsed_ms,1),
        "total_detections":len(detections),"class_counts":dict(Counter(d["label"] for d in detections).most_common()),
        "detections":detections}

def render_metrics(filtered, elapsed_ms):
    pc = sum(1 for d in filtered if d["is_person"])
    oc = sum(1 for d in filtered if not d["is_person"])
    uc = len(set(d["label"] for d in filtered))
    st.markdown(f'<div class="metric-row">'
        f'<div class="metric-card"><div class="label">Total</div><div class="value purple">{len(filtered)}</div></div>'
        f'<div class="metric-card"><div class="label">People</div><div class="value green">{pc}</div></div>'
        f'<div class="metric-card"><div class="label">Objects</div><div class="value blue">{oc}</div></div>'
        f'<div class="metric-card"><div class="label">Classes</div><div class="value amber">{uc}</div></div>'
        f'<div class="metric-card"><div class="label">Inference</div><div class="value purple">{elapsed_ms:.0f}ms</div></div>'
        f'</div>', unsafe_allow_html=True)

def render_detection_list(filtered):
    if not filtered:
        st.info("No detections match the current filter.")
        return
    counts = Counter(d["label"] for d in filtered)
    for cls_name, cnt in counts.most_common():
        bc = "badge-person" if cls_name=="person" else "badge-object"
        st.markdown(f'<div class="det-item"><span>{cls_name}</span><span class="det-badge {bc}">×{cnt}</span></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("##### Per-Detection Details")
    for det in filtered:
        bc = "badge-person" if det["is_person"] else "badge-object"
        tag = "👤" if det["is_person"] else "📦"
        b = det["bbox"]
        st.markdown(f'<div class="det-item"><span>{tag} {det["label"]} — {det["confidence"]:.0%}</span>'
            f'<span class="det-badge {bc}">[{b["x1"]},{b["y1"]}]-[{b["x2"]},{b["y2"]}]</span></div>', unsafe_allow_html=True)

# ── Session State Init ──
# (webcam state is now managed by streamlit-webrtc)

# ── Sidebar ──
with st.sidebar:
    st.markdown("## ⚙️ Detection Controls")
    st.markdown("---")

    input_mode = st.radio("📡 Input Mode", ["📤 Upload Image", "📹 Live Webcam"], index=0)

    st.markdown("---")
    model_size = st.selectbox("🧠 Model Size", ["yolov8n","yolov8s","yolov8m"], index=0,
        help="Nano (fast) → Medium (accurate)")
    st.markdown("---")
    confidence_threshold = st.slider("🎯 Confidence Threshold", 0.10, 0.95, 0.40, 0.05)
    st.markdown("---")
    filter_mode = st.radio("🔎 Filter", ["All Detections","People Only","Objects Only"], index=0)
    st.markdown("---")
    show_json = st.checkbox("📋 Show JSON Output", value=False)

    if input_mode == "📹 Live Webcam":
        st.markdown("---")
        st.markdown("### 📹 Webcam Settings")
        st.markdown("<small style='color:rgba(255,255,255,.45);'>Camera uses your <strong>browser webcam</strong> via WebRTC</small>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='text-align:center;color:rgba(255,255,255,.35);font-size:.75rem;'>YOLOv8 + COCO · 80 classes</div>", unsafe_allow_html=True)

# ── Header ──
st.markdown('<div class="hero-header"><h1>🔍 Object Detection Studio</h1>'
    '<p>Multi-class detection with YOLOv8 · Upload Image or Live Webcam · Scene understanding</p></div>', unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODE 1: UPLOAD IMAGE (preserved exactly)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if input_mode == "📤 Upload Image":
    uploaded_file = st.file_uploader("📤 Upload an image for detection", type=["jpg","jpeg","png","bmp","webp"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image_bgr is None:
            st.error("❌ Failed to load image."); st.stop()

        with st.spinner(f"Loading {model_size} model..."):
            model = load_model(model_size)
        with st.spinner("🔍 Detecting objects..."):
            all_detections, elapsed_ms = run_detection(model, image_bgr, confidence_threshold)

        # Scene classification (Places365 + COCO object fusion)
        with st.spinner("🌍 Analyzing scene (first run downloads Places365 model ~44MB)..."):
            scene_result = classify_scene(all_detections, image_bgr)
        scene_overlay = get_scene_overlay_text(scene_result)

        filtered = filter_detections(all_detections, filter_mode)
        render_metrics(filtered, elapsed_ms)

        # Scene classification card
        render_scene_card(scene_result)

        # Object summary
        summary = generate_scene_summary(filtered)
        st.markdown(f'<div class="scene-box">🌄 <strong>Object Summary:</strong> {summary}</div>', unsafe_allow_html=True)

        col_img, col_list = st.columns([3, 2])
        with col_img:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🖼️ Annotated Image")
            annotated = annotate_image(image_bgr, filtered, scene_overlay)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col_list:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 📋 Detection List")
            render_detection_list(filtered)
            st.markdown('</div>', unsafe_allow_html=True)

        if show_json:
            st.markdown("#### 📊 Structured Output (JSON)")
            structured = build_structured_output(filtered, elapsed_ms, image_bgr.shape)
            structured["scene_classification"] = scene_result
            st.markdown(f'<div class="json-block"><pre>{json.dumps(structured, indent=2)}</pre></div>', unsafe_allow_html=True)

        with st.expander("🔄 Compare Original vs. Annotated", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Original**"); st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            with c2:
                st.markdown("**Annotated**"); st.image(annotated_rgb, use_container_width=True)
    else:
        st.markdown('<div class="glass-card" style="text-align:center;padding:4rem 2rem;">'
            '<p style="font-size:3rem;margin-bottom:.5rem;">📷</p>'
            '<p style="color:rgba(255,255,255,.5);font-size:1.1rem;">Upload an image above to start detecting objects</p>'
            '<p style="color:rgba(255,255,255,.3);font-size:.85rem;">Supports JPG, PNG, BMP, WEBP • YOLOv8 • 80 COCO classes</p></div>', unsafe_allow_html=True)
        st.markdown("---")
        f1,f2,f3 = st.columns(3)
        with f1:
            st.markdown('<div class="glass-card" style="text-align:center;"><p style="font-size:2rem;">🎯</p>'
                '<p style="color:#a78bfa;font-weight:600;">Multi-Class Detection</p>'
                '<p style="color:rgba(255,255,255,.4);font-size:.85rem;">80 object classes from COCO dataset</p></div>', unsafe_allow_html=True)
        with f2:
            st.markdown('<div class="glass-card" style="text-align:center;"><p style="font-size:2rem;">👤</p>'
                '<p style="color:#34d399;font-weight:600;">Person Detection</p>'
                '<p style="color:rgba(255,255,255,.4);font-size:.85rem;">Dedicated person identification</p></div>', unsafe_allow_html=True)
        with f3:
            st.markdown('<div class="glass-card" style="text-align:center;"><p style="font-size:2rem;">🌄</p>'
                '<p style="color:#60a5fa;font-weight:600;">Scene Understanding</p>'
                '<p style="color:rgba(255,255,255,.4);font-size:.85rem;">Auto-generated scene summaries</p></div>', unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODE 2: LIVE WEBCAM (streamlit-webrtc)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

elif input_mode == "📹 Live Webcam":

    # ── Video Processor class for per-frame YOLO inference ──
    # NOTE: Places365 scene classification is DISABLED in webcam mode
    #       to keep CPU usage low on Streamlit Cloud.  Scene analysis
    #       remains fully available in Upload Image mode.
    class YOLOv8VideoProcessor(VideoProcessorBase):
        """Lightweight webcam processor — YOLOv8n detection only.
        No Places365 scene classification to maximise FPS on Cloud."""

        def __init__(self):
            # Always use nano model for webcam (fastest inference)
            self._model = load_model("yolov8n")
            self._confidence = confidence_threshold
            self._filter_mode = filter_mode
            self._lock = threading.Lock()
            # Shared results for the main thread
            self.result_detections: list = []
            self.result_filtered: list = []
            self.result_elapsed_ms: float = 0.0
            self._frame_count = 0
            # Cache last annotated frame for skipped frames
            self._last_annotated = None

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            """Called for every incoming video frame from the browser webcam."""
            img = frame.to_ndarray(format="bgr24")
            self._frame_count += 1

            # Resize for performance (480px wide — lighter than 640)
            h, w = img.shape[:2]
            target_w = 480
            scale = target_w / w
            img_resized = cv2.resize(img, (target_w, int(h * scale)))

            # Run YOLO detection on every 3rd frame to maximise FPS
            if self._frame_count % 3 == 0:
                detections, elapsed_ms = run_detection(
                    self._model, img_resized, self._confidence
                )
                filtered = filter_detections(detections, self._filter_mode)

                # No scene classification — set to None for performance
                scene_result = None
                scene_overlay = None

                annotated = annotate_image(img_resized, filtered, scene_overlay)

                # Share results with the main Streamlit thread
                with self._lock:
                    self.result_detections = detections
                    self.result_filtered = filtered
                    self.result_elapsed_ms = elapsed_ms
                self._last_annotated = annotated
            else:
                # On skipped frames, re-draw with cached detections
                with self._lock:
                    filtered = list(self.result_filtered)
                if filtered:
                    annotated = annotate_image(img_resized, filtered, None)
                elif self._last_annotated is not None:
                    annotated = self._last_annotated
                else:
                    annotated = img_resized

            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    # ── Instruction banner ──
    st.markdown(
        '<div class="webcam-status webcam-live">'
        '📹 Allow browser camera access when prompted · Detection runs in real-time'
        '</div>', unsafe_allow_html=True)

    # ── TURN server config for Streamlit Cloud (NAT traversal) ──
    RTC_CONFIGURATION = {
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }

    # ── Launch the WebRTC streamer ──
    ctx = webrtc_streamer(
        key="yolov8-live-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=YOLOv8VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # ── Display live metrics & detection list below the video ──
    if ctx.state.playing and ctx.video_processor:
        processor = ctx.video_processor
        # Read shared results from the processor (no scene data in webcam mode)
        with processor._lock:
            filtered = list(processor.result_filtered)
            elapsed_ms = processor.result_elapsed_ms

        if filtered:
            render_metrics(filtered, elapsed_ms)
            summary = generate_scene_summary(filtered)
            st.markdown(
                f'<div class="scene-box">🌄 <strong>Object Summary:</strong> {summary}</div>',
                unsafe_allow_html=True)
            with st.expander("📋 Detection List", expanded=False):
                render_detection_list(filtered)

        st.markdown(
            '<div style="text-align:center;color:rgba(255,255,255,.35);font-size:.8rem;margin-top:.5rem;">'
            'Detections update in real-time · Scene classification available in Upload mode</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="glass-card" style="text-align:center;padding:4rem 2rem;">'
            '<p style="font-size:3rem;margin-bottom:.5rem;">📹</p>'
            '<p style="color:rgba(255,255,255,.5);font-size:1.1rem;">'
            'Click <strong>START</strong> above to begin live detection</p>'
            '<p style="color:rgba(255,255,255,.3);font-size:.85rem;">'
            'Uses your browser webcam via WebRTC · YOLOv8n · 80 COCO classes</p></div>',
            unsafe_allow_html=True)
