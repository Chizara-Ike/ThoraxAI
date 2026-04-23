import re
import uuid
import datetime
from io import BytesIO
from pathlib import Path

import cv2
import gdown
import numpy as np
import pandas as pd
import plotly.express as px
import pydicom
import streamlit as st
import tensorflow as tf
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import Flowable
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Thorax AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model_stage3_targeted.h5"
IMG_SIZE = (224, 224)

try:
    ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    ANTHROPIC_API_KEY = ""

@st.cache_resource
def load_model_cached():
    model_dir = BASE_DIR / "models"
    model_dir.mkdir(exist_ok=True)

    if not MODEL_PATH.exists():
        url = "https://drive.google.com/uc?id=1yX70g8IUJluqSd5uIgOv8UQ4HOle1x5U"
        try:
            gdown.download(url, str(MODEL_PATH), quiet=False)
        except Exception as e:
            raise RuntimeError(f"Model download failed: {e}")

    return load_model(MODEL_PATH, compile=False)

CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax"
]

SEVERITY = {
    "Pneumothorax": "high", "Pneumonia": "high", "Edema": "high",
    "Cardiomegaly": "med", "Consolidation": "med", "Mass": "med",
    "Effusion": "med", "Emphysema": "med",
    "Atelectasis": "low", "Fibrosis": "low", "Hernia": "low",
    "Infiltration": "low", "Nodule": "low", "Pleural_Thickening": "low",
}

LOCATION_HINTS = {
    "Atelectasis": "lower lobe regions",
    "Cardiomegaly": "central cardiac silhouette",
    "Consolidation": "parenchymal lung fields",
    "Edema": "bilateral perihilar and basal regions",
    "Effusion": "pleural spaces and costophrenic angles",
    "Emphysema": "bilateral upper lung fields",
    "Fibrosis": "basal and peripheral lung fields",
    "Hernia": "diaphragmatic region",
    "Infiltration": "mid and lower lung zones",
    "Mass": "pulmonary parenchyma",
    "Nodule": "pulmonary parenchyma",
    "Pleural_Thickening": "pleural margins",
    "Pneumonia": "lobar or segmental lung fields",
    "Pneumothorax": "ipsilateral pleural space and lung apex",
}

SEV_LABEL = {"high": "HIGH", "med": "MODERATE", "low": "LOW"}
SYMPTOMS = ["Cough", "Chest Pain", "Shortness of Breath", "Fever", "Weight Loss", "Fatigue"]
LUNG_DISEASE_OPTIONS = ["None", "Asthma", "COPD", "Tuberculosis", "Interstitial Lung Disease", "Previous Pneumonia", "Other"]

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
def ensure_state():
    defaults = {
        "theme": "light",
        "current_step": 1,
        "history": [],
        "ai_explanations": {},
        "patients": {},
        "active_patient_id": None,
        "analysis_results": {},
        "report_edits": {},
        "selected_patient_for_dashboard": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

ensure_state()

# ─────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────
def inject_css(theme: str):
    if theme == "dark":
        bg = "#0b1220"
        surface = "rgba(16, 24, 40, 0.78)"
        surface_2 = "rgba(15, 23, 42, 0.9)"
        card = "#111827"
        border = "rgba(148, 163, 184, 0.18)"
        text = "#ffffff"
        muted = "#94a3b8"
        accent = "#38bdf8"
        accent_2 = "#14b8a6"
        soft = "rgba(56, 189, 248, 0.08)"
        danger = "#f97316"
        success = "#22c55e"
        warning = "#f59e0b"
    else:
        bg = "#f4f8fc"
        surface = "rgba(255,255,255,0.82)"
        surface_2 = "rgba(255,255,255,0.94)"
        card = "#ffffff"
        border = "rgba(15, 23, 42, 0.08)"
        text = "#0f172a"
        muted = "#475569"
        accent = "#0f766e"
        accent_2 = "#2563eb"
        soft = "rgba(37, 99, 235, 0.06)"
        danger = "#dc2626"
        success = "#16a34a"
        warning = "#d97706"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        color: {text};
    }}

    .stApp {{
        background:
            radial-gradient(circle at top right, rgba(37,99,235,0.08), transparent 26%),
            radial-gradient(circle at top left, rgba(20,184,166,0.08), transparent 24%),
            linear-gradient(180deg, {bg} 0%, {bg} 100%);
    }}

    [data-testid="stHeader"] {{
        background: transparent;
    }}

    [data-testid="stSidebar"] {{
        background: {surface_2};
        border-right: 1px solid {border};
    }}

    .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }}

    .hero-card, .glass-card, .step-card, .metric-card, .summary-card {{
        background: {surface};
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid {border};
        border-radius: 22px;
        box-shadow: 0 10px 30px rgba(2, 6, 23, 0.06);
    }}

    .hero-card {{ padding: 1.4rem 1.4rem 1.2rem 1.4rem; margin-bottom: 1rem; }}
    .glass-card {{ padding: 1.2rem; margin-bottom: 1rem; }}
    .step-card {{ padding: 1rem 1.15rem; }}
    .metric-card {{ padding: 1rem 1rem 0.9rem 1rem; height: 100%; }}
    .summary-card {{ padding: 1rem 1.1rem; }}

    .eyebrow {{
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {accent};
        margin-bottom: 0.35rem;
    }}

    .hero-title {{
        font-size: 2rem;
        line-height: 1.1;
        font-weight: 800;
        color: {text};
        margin: 0 0 0.3rem 0;
    }}

    .hero-subtitle {{
        font-size: 0.98rem;
        line-height: 1.6;
        color: {muted};
        margin-bottom: 0.85rem;
    }}

    .status-chip {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: {soft};
        color: {text};
        border: 1px solid {border};
        border-radius: 999px;
        padding: 0.38rem 0.7rem;
        margin-right: 0.45rem;
        margin-bottom: 0.4rem;
        font-size: 0.78rem;
        font-weight: 600;
    }}

    .stepper {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.85rem;
        margin: 1rem 0 1.2rem 0;
    }}

    .step-title {{ font-weight: 700; font-size: 0.9rem; color: {text}; }}
    .step-caption {{ font-size: 0.78rem; color: {muted}; margin-top: 0.15rem; }}
    .step-index {{
        width: 34px; height: 34px; border-radius: 50%; display:flex; align-items:center; justify-content:center;
        font-weight: 800; font-size: 0.9rem; margin-bottom: 0.7rem; border:1px solid {border};
        background: {soft}; color: {accent_2};
    }}
    .step-active {{ outline: 2px solid rgba(37,99,235,0.14); }}

    .section-title {{ font-size: 1.06rem; font-weight: 800; color: {text}; margin-bottom: 0.2rem; }}
    .section-sub {{ font-size: 0.88rem; color: {muted}; margin-bottom: 1rem; }}

    .pill {{
        display:inline-block; padding:0.34rem 0.62rem; border-radius:999px; font-size:0.74rem; font-weight:700;
        margin:0.15rem 0.25rem 0.15rem 0; border:1px solid {border}; background:{soft}; color:{text};
    }}
    .pill.high {{ background: rgba(239,68,68,0.1); color: {danger}; }}
    .pill.med {{ background: rgba(245,158,11,0.12); color: {warning}; }}
    .pill.low {{ background: rgba(34,197,94,0.12); color: {success}; }}

    .metric-label {{ font-size: 0.8rem; color: {muted}; margin-bottom: 0.25rem; }}
    .metric-value {{ font-size: 1.45rem; font-weight: 800; color: {text}; }}

    div[data-testid="stButton"] > button,
    div[data-testid="stDownloadButton"] > button {{
        border-radius: 14px;
        border: 1px solid {border};
        font-weight: 700;
        padding: 0.65rem 1rem;
        background: linear-gradient(180deg, {card} 0%, {surface_2} 100%);
        color: {text};
    }}

    div[data-testid="stButton"] > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {{
        border-color: rgba(37,99,235,0.25);
        color: {accent_2};
    }}

    div[data-testid="stForm"] {{
        border: none !important;
    }}

    .footer-note {{
        margin-top: 1.3rem; padding: 0.9rem 1rem; border-radius: 18px; border: 1px solid {border};
        background: {surface}; color: {muted}; font-size: 0.84rem;
    }}
    </style>
    """, unsafe_allow_html=True)

inject_css(st.session_state.theme)

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
try:
    model = load_model_cached()
    model_loaded = True
    model_error = ""
except Exception as e:
    model_loaded = False
    model_error = str(e)
    model = None

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def safe_key(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", text)

def preprocess_image(image: Image.Image):
    img = np.array(image.resize(IMG_SIZE).convert("RGB")) / 255.0
    return np.expand_dims(img, axis=0).astype(np.float32)

def find_last_conv_layer(mdl):
    for layer in reversed(mdl.layers):
        try:
            if len(layer.output_shape) == 4:
                return layer.name
        except Exception:
            continue
    raise ValueError("No convolutional layer found.")

def make_gradcam_heatmap(img_array, mdl, last_conv, threshold):
    img_t = tf.constant(img_array, dtype=tf.float32)
    first_error = None
    try:
        with tf.GradientTape(persistent=True) as tape:
            iv = tf.Variable(img_t, trainable=False)
            tape.watch(iv)
            x = iv
            conv_out = None
            for layer in mdl.layers:
                try:
                    x = layer(x, training=False)
                    if layer.name == last_conv:
                        conv_out = x
                        tape.watch(conv_out)
                except Exception:
                    continue
            if conv_out is None:
                raise ValueError(f"Layer {last_conv!r} not reached.")
            preds = mdl(iv, training=False)
            ci = tf.where(preds[0] > threshold)
            cidx = int(tf.argmax(preds[0])) if tf.shape(ci)[0] == 0 else int(ci[0][0])
            loss = preds[:, cidx]
        grads = tape.gradient(loss, conv_out)
        del tape
        if grads is None:
            raise ValueError("Gradients None.")
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy(), cidx, last_conv
    except Exception as e:
        first_error = e

    try:
        iv = tf.Variable(img_t, trainable=False)
        with tf.GradientTape() as tape:
            tape.watch(iv)
            preds = mdl(iv, training=False)
            ci = tf.where(preds[0] > threshold)
            cidx = int(tf.argmax(preds[0])) if tf.shape(ci)[0] == 0 else int(ci[0][0])
            loss = preds[:, cidx]
        grads = tape.gradient(loss, iv)
        if grads is None:
            raise ValueError("Input grads None.")
        heatmap = tf.reduce_max(tf.abs(grads[0]), axis=-1)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy(), cidx, "input_saliency"
    except Exception as e2:
        raise RuntimeError(f"Grad-CAM failed. S1: {first_error} | S2: {e2}")

def overlay_heatmap(heatmap, image, alpha=0.42):
    h, w = image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    hmap = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_VIRIDIS)
    combined = np.clip(hmap * alpha + image.astype(np.float32), 0, 255).astype(np.uint8)
    return combined, heatmap_resized

def load_dicom(uploaded):
    ds = pydicom.dcmread(uploaded)
    arr = ds.pixel_array.astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
    arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(arr)

def img_to_bytes(arr):
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()

def get_attention_bbox(heatmap_resized, threshold_ratio=0.65):
    mask = (heatmap_resized >= float(np.max(heatmap_resized)) * threshold_ratio).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return {"x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": int(w * h)}

def severity_color_key(label):
    return SEVERITY.get(label, "low")

def build_ai_prompt(detected, patient, threshold):
    age = patient.get("age", "Unknown")
    gender = patient.get("gender", "Unknown")
    smoker = patient.get("smoker", "Unknown")
    symptoms = ", ".join(patient.get("symptoms", [])) or "None reported"
    diseases = ", ".join(patient.get("known_lung_disease", [])) or "None reported"
    occupation = patient.get("occupation", "Not specified")
    pack_years = patient.get("pack_years", 0)
    spo2 = patient.get("spo2", "Not recorded")
    rr = patient.get("resp_rate", "Not recorded")

    if detected:
        lines = "\n".join(
            f"- {l} (confidence: {p*100:.1f}%, severity: {SEVERITY.get(l,'low')}, typical location: {LOCATION_HINTS.get(l,'lung fields')})"
            for l, p in detected
        )
    else:
        lines = "- No significant pathology detected above the confidence threshold."

    return f"""You are a board-certified radiologist providing clinical decision support.

An AI model analysed a chest X-ray for a {age}-year-old {gender} patient.
Smoking status: {smoker}
Pack-years: {pack_years}
Occupation / exposure context: {occupation}
Known lung disease history: {diseases}
Presenting symptoms: {symptoms}
SpO₂: {spo2}
Respiratory rate: {rr}
Detection threshold: {threshold:.0%}

AI-detected findings:
{lines}

Write a structured clinical interpretation using EXACTLY these section headers on their own lines:

**Imaging Findings**
For each detected pathology, describe the radiographic appearance in clear clinical language, including location and key visual characteristics. Avoid unnecessary repetition.

**Disease Indicators**
For each finding, explain what the imaging feature suggests and state the most likely diagnosis using full medical terminology.

**Aetiology**
For each diagnosis, list the most relevant underlying causes, prioritizing those appropriate for this patient’s age, gender, smoking status, symptoms, and available history.

**Clinical Significance**
Provide 2–3 sentences summarizing the overall clinical picture.
Include whether findings are mild, moderate, or severe, level of concern, and relevance to this patient’s demographics and symptoms.

**Disclaimer**
This is an AI-assisted interpretation and must be confirmed by a qualified radiologist and treating physician before clinical decisions are made.

Write in full sentences only. No bullet points. Avoid redundancy. Prioritize clinical clarity and accuracy."""

def stream_ai_explanation(detected, patient, threshold):
    try:
        import anthropic
    except ImportError:
        yield "⚠ `anthropic` not installed."
        return

    if not ANTHROPIC_API_KEY:
        yield "⚠ ANTHROPIC_API_KEY not set."
        return

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    system = "You are a board-certified radiologist AI assistant. Follow the prompt precisely and stay clinically clear."
    try:
        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=1400,
            system=system,
            messages=[{"role": "user", "content": build_ai_prompt(detected, patient, threshold)}],
        ) as stream:
            for text in stream.text_stream:
                yield text
    except Exception as e:
        yield f"\n\n⚠ AI error: {e}"

def get_ai_explanation_sync(detected, patient, threshold):
    try:
        import anthropic
    except ImportError:
        return "AI unavailable: `anthropic` not installed."

    if not ANTHROPIC_API_KEY:
        return "AI unavailable: ANTHROPIC_API_KEY not set."

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    system = "You are a board-certified radiologist AI assistant. Follow the prompt precisely and stay clinically clear."
    try:
        resp = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1400,
            system=system,
            messages=[{"role": "user", "content": build_ai_prompt(detected, patient, threshold)}],
        )
        return resp.content[0].text
    except Exception as e:
        return f"AI analysis could not be generated: {e}"

class ConfidenceBarChart(Flowable):
    def __init__(self, names, probs, threshold, width=490):
        Flowable.__init__(self)
        self.names = names
        self.probs = probs
        self.threshold = threshold
        self.label_w = 152
        self.val_w = 46
        self.bar_max_w = width - self.label_w - self.val_w - 8
        self.row_h = 18
        self.width = width
        self.height = self.row_h * len(names) + 6

    def draw(self):
        max_p = max(self.probs) if max(self.probs) > 0 else 1.0
        pairs = sorted(zip(self.names, self.probs), key=lambda x: -x[1])
        for i, (name, prob) in enumerate(pairs):
            y = self.height - (i + 1) * self.row_h + 2
            above = prob >= self.threshold
            if i % 2 == 0:
                self.canv.setFillColor(colors.HexColor("#f8fafc"))
                self.canv.rect(0, y - 2, self.width, self.row_h, fill=1, stroke=0)
            self.canv.setFont("Helvetica-Bold" if above else "Helvetica", 7.5)
            self.canv.setFillColor(colors.HexColor("#1e3a5f") if above else colors.HexColor("#6b7280"))
            self.canv.drawString(2, y + 4, name.replace("_", " "))
            self.canv.setFillColor(colors.HexColor("#e2e8f0"))
            self.canv.roundRect(self.label_w, y + 3, self.bar_max_w, 10, 3, fill=1, stroke=0)
            bar_len = max((prob / max_p) * self.bar_max_w, 2)
            self.canv.setFillColor(colors.HexColor("#1e3a5f") if above else colors.HexColor("#93c5fd"))
            self.canv.roundRect(self.label_w, y + 3, bar_len, 10, 3, fill=1, stroke=0)
            self.canv.setFont("Helvetica-Bold" if above else "Helvetica", 7)
            self.canv.setFillColor(colors.HexColor("#1e3a5f") if above else colors.HexColor("#64748b"))
            self.canv.drawString(self.label_w + self.bar_max_w + 5, y + 4, f"{prob*100:.1f}%")
        if max_p > 0:
            mx = self.label_w + (self.threshold / max_p) * self.bar_max_w
            self.canv.setStrokeColor(colors.HexColor("#dc2626"))
            self.canv.setLineWidth(0.8)
            self.canv.setDash(3, 3)
            self.canv.line(mx, 0, mx, self.height)
            self.canv.setDash()
            self.canv.setFont("Helvetica", 6)
            self.canv.setFillColor(colors.HexColor("#dc2626"))
            self.canv.drawString(mx + 2, self.height - 8, f"Threshold {self.threshold:.0%}")

def _sec(num, title, styles):
    s = ParagraphStyle("SecHdr", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=9,
                       textColor=colors.white, backColor=colors.HexColor("#1e3a5f"),
                       borderPad=(5, 8, 5, 8), leading=14, spaceAfter=6)
    return Paragraph(f"{num}.  {title.upper()}", s)

def _parse_ai(ai_text, styles):
    sub_s = ParagraphStyle("AIH", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=9,
                           textColor=colors.HexColor("#1e3a5f"), spaceBefore=8, spaceAfter=3, leading=13)
    bod_s = ParagraphStyle("AIB", parent=styles["Normal"], fontName="Helvetica", fontSize=8.5,
                           textColor=colors.HexColor("#374151"), leading=13, spaceAfter=4)
    elements = []
    if not ai_text or ai_text.startswith("AI unavailable") or ai_text.startswith("AI analysis"):
        elements.append(Paragraph(ai_text or "Not available.", bod_s))
        return elements
    buf = []
    for raw in ai_text.split("\n"):
        line = raw.strip()
        if not line:
            if buf:
                elements.append(Paragraph(" ".join(buf), bod_s))
                buf = []
            continue
        if line.startswith("**") and line.endswith("**") and len(line) > 4:
            if buf:
                elements.append(Paragraph(" ".join(buf), bod_s))
                buf = []
            elements.append(Paragraph(line[2:-2], sub_s))
        else:
            buf.append(line.replace("**", ""))
    if buf:
        elements.append(Paragraph(" ".join(buf), bod_s))
    return elements

def _rl(arr, w, h):
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    buf.seek(0)
    return RLImage(buf, width=w, height=h)

def build_pdf(patient, detected, preds, original_img, cam_img, threshold, edited_ai_text=None):
    buffer = BytesIO()
    page_w, _ = A4
    lm = rm = 35
    body_w = page_w - lm - rm
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=rm, leftMargin=lm, topMargin=28, bottomMargin=28)
    styles = getSampleStyleSheet()
    el = []

    now = datetime.datetime.now()
    report_id = f"AST-{now.strftime('%Y%m%d-%H%M%S')}"
    gen_str = now.strftime("%d %B %Y, %H:%M")

    h1s = ParagraphStyle("H1", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=11,
                         textColor=colors.white, alignment=TA_CENTER, leading=16)
    h2s = ParagraphStyle("H2", parent=styles["Normal"], fontName="Helvetica", fontSize=8,
                         textColor=colors.HexColor("#93c5fd"), alignment=TA_CENTER, leading=12)
    hdr = Table([
        [Paragraph("THORAX AI", h1s)],
        [Paragraph("Intelligent Clinical Decision Support System for Thoracic Pathology Diagnosis", h2s)],
        [Paragraph("Radiology and Clinical Review Report", h2s)],
    ], colWidths=[body_w])
    hdr.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#1e3a5f")),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    el.extend([hdr, Spacer(1, 8)])

    title_s = ParagraphStyle("T", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=14,
                             textColor=colors.HexColor("#1e3a5f"), alignment=TA_CENTER, leading=20)
    meta_s = ParagraphStyle("M", parent=styles["Normal"], fontName="Helvetica", fontSize=8,
                            textColor=colors.HexColor("#6b7280"), alignment=TA_CENTER, leading=12)
    el.append(Paragraph("THORACIC PATHOLOGY DIAGNOSIS REPORT", title_s))
    el.append(Spacer(1, 4))
    el.append(Paragraph(f"Report ID: {report_id} | Generated: {gen_str}", meta_s))
    el.append(HRFlowable(width=body_w, thickness=1.5, color=colors.HexColor("#1e3a5f"), spaceAfter=10))

    el.append(_sec("1", "Patient Demographics and Clinical Context", styles))
    ks = ParagraphStyle("K", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=8.5, textColor=colors.HexColor("#1e3a5f"))
    vs = ParagraphStyle("V", parent=styles["Normal"], fontName="Helvetica", fontSize=8.5, textColor=colors.HexColor("#374151"))
    cw = body_w / 4
    lung_history = ", ".join(patient.get("known_lung_disease", [])) or "None"
    symptoms = ", ".join(patient.get("symptoms", [])) or "None"
    demo = Table([
        [Paragraph("Patient ID", ks), Paragraph(patient.get("patient_id", "N/A"), vs), Paragraph("Age", ks), Paragraph(str(patient.get("age", "N/A")), vs)],
        [Paragraph("Full Name", ks), Paragraph(patient.get("full_name", "N/A"), vs), Paragraph("Gender", ks), Paragraph(patient.get("gender", "N/A"), vs)],
        [Paragraph("Smoking Status", ks), Paragraph(patient.get("smoker", "N/A"), vs), Paragraph("Pack-Years", ks), Paragraph(str(patient.get("pack_years", "N/A")), vs)],
        [Paragraph("Occupation", ks), Paragraph(patient.get("occupation", "N/A"), vs), Paragraph("Thoracic Surgery", ks), Paragraph("Yes" if patient.get("thoracic_surgery") else "No", vs)],
        [Paragraph("Known Lung Disease", ks), Paragraph(lung_history, vs), Paragraph("Symptoms", ks), Paragraph(symptoms, vs)],
        [Paragraph("SpO₂", ks), Paragraph(str(patient.get("spo2", "N/A")), vs), Paragraph("Respiratory Rate", ks), Paragraph(str(patient.get("resp_rate", "N/A")), vs)],
    ], colWidths=[cw * 0.7, cw * 1.3, cw * 0.7, cw * 1.3])
    demo.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#f0f4ff"), colors.white]),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
    ]))
    el.extend([demo, Spacer(1, 10)])

    el.append(_sec("2", "Clinical Findings Summary", styles))
    sev_colors = {
        "high": (colors.HexColor("#fee2e2"), colors.HexColor("#991b1b")),
        "med": (colors.HexColor("#fef3c7"), colors.HexColor("#92400e")),
        "low": (colors.HexColor("#dcfce7"), colors.HexColor("#166534")),
    }
    if detected:
        th_s = ParagraphStyle("TH", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=8, textColor=colors.white)
        rows = [[Paragraph(t, th_s) for t in ["PATHOLOGY", "CONFIDENCE", "SEVERITY", "TYPICAL LOCATION"]]]
        for label, prob in sorted(detected, key=lambda x: -x[1]):
            sev = SEVERITY.get(label, "low")
            bg, fg = sev_colors[sev]
            sev_s = ParagraphStyle(f"S{label}", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=8, textColor=fg, backColor=bg, borderPad=3)
            rows.append([
                Paragraph(label.replace("_", " "), ParagraphStyle("FN", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=8.5, textColor=colors.HexColor("#1e3a5f"))),
                Paragraph(f"{prob*100:.2f}%", ParagraphStyle("FC", parent=styles["Normal"], fontName="Helvetica", fontSize=8.5, textColor=colors.HexColor("#374151"), alignment=TA_CENTER)),
                Paragraph(SEV_LABEL.get(sev, sev.upper()), sev_s),
                Paragraph(LOCATION_HINTS.get(label, "—"), ParagraphStyle("FL", parent=styles["Normal"], fontName="Helvetica", fontSize=8, textColor=colors.HexColor("#374151"))),
            ])
        ft = Table(rows, colWidths=[body_w * 0.24, body_w * 0.14, body_w * 0.14, body_w * 0.48])
        ft.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        el.append(ft)
    else:
        ok_s = ParagraphStyle("OK", parent=styles["Normal"], fontName="Helvetica", fontSize=8.5, textColor=colors.HexColor("#166534"), backColor=colors.HexColor("#dcfce7"), borderPad=6)
        el.append(Paragraph("No significant pathology detected above the configured confidence threshold.", ok_s))
    el.append(Spacer(1, 10))

    el.append(_sec("3", "Confidence Distribution: All Pathologies", styles))
    cap_s = ParagraphStyle("Cap", parent=styles["Normal"], fontName="Helvetica-Oblique", fontSize=7.5, textColor=colors.HexColor("#6b7280"), spaceAfter=6)
    el.append(Paragraph(
        f"Figure 1. Predicted confidence scores for all 14 pathology classes. Dark navy bars indicate detections at or above the {threshold:.0%} threshold. The dashed red line marks the threshold boundary.",
        cap_s,
    ))
    el.append(ConfidenceBarChart(CLASS_NAMES, [float(p) for p in preds], threshold, width=int(body_w)))
    el.append(Spacer(1, 12))

    el.append(PageBreak())
    el.append(_sec("4", "Radiological Images", styles))
    img_cap_s = ParagraphStyle("IC", parent=styles["Normal"], fontName="Helvetica-Oblique", fontSize=7.5, textColor=colors.HexColor("#374151"), alignment=TA_CENTER)
    col_lbl_s = ParagraphStyle("CL", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=8, textColor=colors.HexColor("#1e3a5f"), alignment=TA_CENTER)
    img_w = (body_w / 2) - 10
    it = Table([
        [Paragraph("Original Chest X-ray", col_lbl_s), Paragraph("Grad-CAM Saliency Overlay", col_lbl_s)],
        [_rl(original_img, img_w, img_w), _rl(cam_img, img_w, img_w)],
        [Paragraph("Fig. 2. Original chest X-ray submitted for diagnosis.", img_cap_s), Paragraph("Fig. 3. Saliency overlay highlighting the regions that most influenced the prediction.", img_cap_s)],
    ], colWidths=[body_w / 2, body_w / 2])
    it.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#e2e8f0")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f4ff")),
    ]))
    el.extend([it, Spacer(1, 12)])

    el.append(_sec("5", "AI-assisted Clinical Interpretation", styles))
    ai_meta_s = ParagraphStyle("AIM", parent=styles["Normal"], fontName="Helvetica-Oblique", fontSize=7.5, textColor=colors.HexColor("#6b7280"), spaceAfter=8)
    el.append(Paragraph(f"Auto-generated clinical narrative · {gen_str}", ai_meta_s))
    ai_text = edited_ai_text or get_ai_explanation_sync(detected, patient, threshold)
    ai_elements = _parse_ai(ai_text, styles)
    ai_box = Table([[elem] for elem in ai_elements], colWidths=[body_w - 20])
    ai_box.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#1e3a5f")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    el.extend([ai_box, Spacer(1, 16)])

    disc_title_s = ParagraphStyle("DT", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=7.5, textColor=colors.HexColor("#374151"), spaceAfter=2)
    disc_s = ParagraphStyle("DS", parent=styles["Normal"], fontName="Helvetica", fontSize=7.5, textColor=colors.HexColor("#6b7280"), leading=11, spaceAfter=4)
    ftr_s = ParagraphStyle("FS", parent=styles["Normal"], fontName="Helvetica", fontSize=7, textColor=colors.HexColor("#9ca3af"), alignment=TA_CENTER)
    el.append(HRFlowable(width=body_w, thickness=1, color=colors.HexColor("#1e3a5f"), spaceBefore=4))
    el.append(Paragraph("DISCLAIMER & LIMITATIONS", disc_title_s))
    el.append(Paragraph(
        "This report is generated by an experimental AI-based clinical decision support system intended for research and educational support purposes only. It does not constitute a definitive medical diagnosis. All findings must be reviewed and confirmed by a qualified radiologist or clinician before clinical decisions are made.",
        disc_s,
    ))
    el.append(Spacer(1, 4))
    el.append(Paragraph(f"Thorax AI | Report {report_id}", ftr_s))

    doc.build(el)
    buffer.seek(0)
    return buffer

def plot_image_viewer(image_array, title, bbox=None):
    fig = px.imshow(image_array)
    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=40, b=10),
        dragmode="pan",
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor="x"),
        coloraxis_showscale=False,
    )
    if bbox:
        x0, y0 = bbox["x"], bbox["y"]
        x1, y1 = x0 + bbox["w"], y0 + bbox["h"]
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color="red", width=3), fillcolor="rgba(0,0,0,0)")
        fig.add_annotation(x=x0, y=max(y0 - 12, 5), text="High-attention region", showarrow=False, font=dict(size=12, color="red"), bgcolor="rgba(255,255,255,0.8)")
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True})

def go_to_step(step: int):
    st.session_state.current_step = step

def create_patient_record(form_data):
    patient_id = form_data["patient_id"].strip()
    st.session_state.patients[patient_id] = {
        **form_data,
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "analyses": st.session_state.patients.get(patient_id, {}).get("analyses", []),
    }
    st.session_state.active_patient_id = patient_id
    st.session_state.selected_patient_for_dashboard = patient_id

def current_patient():
    pid = st.session_state.active_patient_id
    if not pid:
        return None
    return st.session_state.patients.get(pid)

def store_analysis(patient_id, result):
    st.session_state.analysis_results[result["analysis_id"]] = result
    st.session_state.patients[patient_id].setdefault("analyses", []).append(result["analysis_id"])
    st.session_state.selected_patient_for_dashboard = patient_id

    if not any(h.get("analysis_id") == result["analysis_id"] for h in st.session_state.history):
        st.session_state.history.append({
            "analysis_id": result["analysis_id"],
            "timestamp": result["timestamp"],
            "patient_id": patient_id,
            "name": st.session_state.patients[patient_id].get("full_name", "—"),
            "file": result["file_name"],
            "detected": ", ".join([l for l, _ in result["detected"]]) if result["detected"] else "None",
            "n_findings": len(result["detected"]),
            "top_conf": f"{result['top_conf']*100:.1f}%",
        })

def header():
    patient = current_patient()
    with st.container():
        st.markdown('<div class="hero-card">', unsafe_allow_html=True)
        c1, c2 = st.columns([3.2, 1.3])
        with c1:
            st.markdown('<div class="eyebrow">Intelligent Clinical Decision Support System for Thoracic Pathology Diagnosis</div>', unsafe_allow_html=True)
            st.markdown('<div class="hero-title">Thorax AI</div>', unsafe_allow_html=True)
            st.markdown('<div class="hero-subtitle">Built for radiologists, clinicians, and medical students.</div>', unsafe_allow_html=True)
            model_chip = "Model ready" if model_loaded else "Model unavailable"
            ai_chip = "AI explanation ready" if ANTHROPIC_API_KEY else "AI explanation unavailable"
            active_chip = f"Active patient: {patient.get('full_name')}" if patient else "No active patient"
            st.markdown(
                f'<span class="status-chip">🧠 {model_chip}</span>'
                f'<span class="status-chip">📝 {ai_chip}</span>'
                f'<span class="status-chip">👤 {active_chip}</span>',
                unsafe_allow_html=True,
            )
        with c2:
            current = st.session_state.theme
            choice = st.radio("Theme Settings", ["light", "dark"], index=0 if current == "light" else 1, horizontal=True)
            if choice != current:
                st.session_state.theme = choice
                st.rerun()
            st.caption("Default theme is light mode.")
        st.markdown('</div>', unsafe_allow_html=True)

def stepper():
    steps = [
        (1, "Patient Intake", "Demographics, symptoms, risk profile"),
        (2, "Imaging & Analysis", "Upload, zoom, detect, explain"),
        (3, "Report & Review", "Edit report, export PDF, compare"),
        (4, "Patient Sessions", "Registry, trends, multi-patient review"),
    ]
    cols = st.columns(4)
    for col, (num, title, caption) in zip(cols, steps):
        extra = " step-active" if st.session_state.current_step == num else ""
        with col:
            st.markdown(f'<div class="step-card{extra}"><div class="step-index">{num}</div><div class="step-title">{title}</div><div class="step-caption">{caption}</div></div>', unsafe_allow_html=True)
            if st.button(f"Open", key=f"step_btn_{num}", use_container_width=True):
                go_to_step(num)

def render_intake_step():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Patient Intake</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Capture the clinical context first so downstream analysis, report generation, and trend review stay tied to the correct patient.</div>', unsafe_allow_html=True)

    with st.form("patient_intake_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            patient_id = st.text_input("Patient ID *", placeholder="e.g. PT-00123")
            full_name = st.text_input("Full Name *", placeholder="Surname, First name")
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            gender = st.selectbox("Gender", ["Male", "Female", "Other / Prefer not to say"])
            smoker = st.selectbox("Smoking Status", ["No", "Yes", "Ex-smoker"])
            pack_years = st.number_input("Pack-years", min_value=0.0, max_value=120.0, value=0.0, step=0.5)
        with c2:
            occupation = st.text_input("Occupation / Exposure Context", placeholder="e.g. Miner, factory worker, office worker")
            known_lung_disease = st.multiselect("Known Lung Disease / Thoracic History", LUNG_DISEASE_OPTIONS, default=[])
            thoracic_surgery = st.toggle("Previous thoracic surgery")
            family_history = st.toggle("Family history relevant to lung / thoracic disease")
            spo2 = st.text_input("SpO₂", placeholder="e.g. 96%")
            resp_rate = st.text_input("Respiratory Rate", placeholder="e.g. 18 breaths/min")

        symptoms = st.multiselect("Presenting Symptoms", SYMPTOMS)
        clinical_notes = st.text_area("Clinical Notes", placeholder="Optional notes for radiology / clinician review")
        threshold = st.slider("Detection Threshold", 0.1, 0.9, 0.5, 0.05, help="Minimum confidence to flag a pathology.")

        submitted = st.form_submit_button("Save Patient & Proceed", use_container_width=True)
        if submitted:
            if not patient_id.strip() or not full_name.strip():
                st.error("Patient ID and Full Name are required.")
            else:
                form_data = {
                    "patient_id": patient_id.strip(),
                    "full_name": full_name.strip(),
                    "age": int(age),
                    "gender": gender,
                    "smoker": smoker,
                    "pack_years": float(pack_years),
                    "occupation": occupation.strip() or "Not specified",
                    "known_lung_disease": known_lung_disease,
                    "thoracic_surgery": thoracic_surgery,
                    "family_history": family_history,
                    "spo2": spo2.strip() or "Not recorded",
                    "resp_rate": resp_rate.strip() or "Not recorded",
                    "symptoms": symptoms,
                    "clinical_notes": clinical_notes.strip(),
                    "threshold": float(threshold),
                }
                create_patient_record(form_data)
                st.success(f"Patient {full_name.strip()} is now active.")
                go_to_step(2)
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.patients:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Quick Patient Switch</div>', unsafe_allow_html=True)
        options = list(st.session_state.patients.keys())
        selected = st.selectbox(
            "Choose active patient",
            options,
            index=options.index(st.session_state.active_patient_id) if st.session_state.active_patient_id in options else 0,
            format_func=lambda x: f"{x} — {st.session_state.patients[x]['full_name']}",
        )
        if selected != st.session_state.active_patient_id:
            st.session_state.active_patient_id = selected
            st.success(f"Active patient switched to {st.session_state.patients[selected]['full_name']}.")
        st.markdown('</div>', unsafe_allow_html=True)

def render_analysis_step():
    patient = current_patient()
    if not patient:
        st.warning("Create or select a patient before analysis.")
        if st.button("Go to Patient Intake"):
            go_to_step(1)
            st.rerun()
        return

    threshold = float(patient.get("threshold", 0.5))
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Imaging & Analysis</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">Active patient: <b>{patient.get("full_name")}</b> · {patient.get("patient_id")} · Threshold {threshold:.0%}</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload X-ray(s)",
        type=["jpg", "jpeg", "png", "dcm"],
        accept_multiple_files=True,
        help="Supports JPEG, PNG, and DICOM files.",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if not uploaded_files:
        st.info("Upload one or more chest X-rays to begin the diagnosis workflow.")
        return

    for file_idx, uploaded_file in enumerate(uploaded_files):
        unique_name = f"{patient['patient_id']}_{file_idx}_{safe_key(uploaded_file.name)}"
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">{uploaded_file.name}</div>', unsafe_allow_html=True)
        st.caption(f"Analysed for {patient['full_name']} · {datetime.datetime.now().strftime('%d %b %Y, %I:%M %p')}")

        try:
            if uploaded_file.name.lower().endswith(".dcm"):
                image = load_dicom(uploaded_file)
                st.caption("DICOM detected. Converted to RGB for visualization and analysis.")
            else:
                image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.markdown('</div>', unsafe_allow_html=True)
            continue

        if not model_loaded:
            st.error(f"Model is not loaded: {model_error}")
            st.markdown('</div>', unsafe_allow_html=True)
            continue

        img_array = preprocess_image(image)
        progress = st.progress(0, text="Preparing image")
        try:
            progress.progress(25, text="Running model inference")
            preds = model.predict(img_array, verbose=0)[0]
            progress.progress(60, text="Generating explainability map")
            last_conv = find_last_conv_layer(model)
            heatmap, class_idx, cam_layer = make_gradcam_heatmap(img_array, model, last_conv, threshold)
            img_np = np.array(image.convert("RGB"))
            cam_img, heatmap_resized = overlay_heatmap(heatmap, img_np)
            bbox = get_attention_bbox(heatmap_resized)
            cam_ok = True
            cam_err = ""
        except Exception as e:
            cam_ok = False
            cam_err = str(e)
            img_np = np.array(image.convert("RGB"))
            cam_img = img_np.copy()
            heatmap_resized = None
            bbox = None
        progress.progress(85, text="Compiling result summary")
        detected = [(CLASS_NAMES[i], float(preds[i])) for i in range(len(CLASS_NAMES)) if preds[i] >= threshold]
        detected_sorted = sorted(detected, key=lambda x: -x[1])
        progress.progress(100, text="Analysis complete")
        progress.empty()

        m1, m2, m3, m4 = st.columns(4)
        metrics = [
            ("Pathologies Found", str(len(detected_sorted))),
            ("Top Confidence", f"{float(np.max(preds))*100:.1f}%"),
            ("Threshold", f"{threshold:.0%}"),
            ("Primary Class", CLASS_NAMES[int(np.argmax(preds))].replace("_", " ")),
        ]
        for col, (label, value) in zip([m1, m2, m3, m4], metrics):
            with col:
                st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

        st.markdown("### Viewer")
        viewer_mode = st.radio(
            f"Viewer mode · {uploaded_file.name}",
            ["Original", "Grad-CAM Overlay", "Side by Side"],
            horizontal=True,
            key=f"viewer_{unique_name}",
        )
        if viewer_mode == "Original":
            plot_image_viewer(img_np, "Original X-ray", None)
        elif viewer_mode == "Grad-CAM Overlay":
            plot_image_viewer(cam_img, f"Overlay · {CLASS_NAMES[class_idx].replace('_', ' ')}", bbox if cam_ok else None)
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                plot_image_viewer(img_np, "Original X-ray", None)
            with col_b:
                plot_image_viewer(cam_img, "Grad-CAM Overlay", bbox if cam_ok else None)

        info_col, highlight_col = st.columns([1.25, 1])
        with info_col:
            st.markdown("### Findings")
            if detected_sorted:
                chips = "".join([f'<span class="pill {severity_color_key(label)}">{label.replace("_", " ")} · {prob*100:.0f}%</span>' for label, prob in detected_sorted])
                st.markdown(chips, unsafe_allow_html=True)
            else:
                st.success("No significant pathology detected above the current threshold.")

            prob_df = pd.DataFrame({
                "Pathology": [c.replace("_", " ") for c in CLASS_NAMES],
                "Probability": [float(p) for p in preds],
            }).sort_values("Probability", ascending=False)
            st.bar_chart(prob_df.set_index("Pathology")["Probability"], use_container_width=True)
            with st.expander("Detailed probability table"):
                st.dataframe(prob_df.style.format({"Probability": "{:.3f}"}), use_container_width=True)

        with highlight_col:
            st.markdown("### Region Highlighting")
            if cam_ok and bbox:
                top_label = detected_sorted[0][0] if detected_sorted else CLASS_NAMES[class_idx]
                st.markdown('<div class="summary-card">', unsafe_allow_html=True)
                st.write(f"**Dominant attention region:** x={bbox['x']}, y={bbox['y']}, w={bbox['w']}, h={bbox['h']}")
                st.write(f"**Associated pathology:** {top_label.replace('_', ' ')}")
                st.write(f"**Suggested anatomical emphasis:** {LOCATION_HINTS.get(top_label, 'Lung field review recommended')}")
                st.write(f"**Grad-CAM source layer:** {cam_layer}")
                st.markdown('</div>', unsafe_allow_html=True)
            elif not cam_ok:
                st.warning(f"Region highlighting unavailable because Grad-CAM failed: {cam_err}")
            else:
                st.info("No strong localized high-attention region was isolated.")

        analysis_id = f"{patient['patient_id']}_{uuid.uuid4().hex[:8]}"
        result_payload = {
            "analysis_id": analysis_id,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "patient_id": patient["patient_id"],
            "file_name": uploaded_file.name,
            "detected": detected_sorted,
            "preds": [float(p) for p in preds],
            "top_conf": float(np.max(preds)) if len(preds) else 0.0,
            "primary_class": CLASS_NAMES[int(np.argmax(preds))],
            "img_np": img_np,
            "cam_img": cam_img,
            "bbox": bbox,
            "threshold": threshold,
            "patient_snapshot": dict(patient),
        }

        if st.button("Save Analysis to Patient Session", key=f"save_analysis_{unique_name}", use_container_width=True):
            store_analysis(patient["patient_id"], result_payload)
            st.success("Analysis saved. It is now available in Report & Review and Patient Sessions.")

        st.download_button(
            "Download Grad-CAM Image",
            data=img_to_bytes(cam_img),
            file_name=f"gradcam_{uploaded_file.name}.png",
            mime="image/png",
            key=f"dl_cam_{unique_name}",
            use_container_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

def render_report_step():
    patient = current_patient()
    if not patient:
        st.warning("Create or select a patient first.")
        return
    analysis_ids = patient.get("analyses", [])
    if not analysis_ids:
        st.info("No saved analyses for this patient yet. Save at least one result from Step 2 first.")
        return

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Report & Review</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Edit the narrative before exporting, compare current and prior scans, and generate a PDF.</div>', unsafe_allow_html=True)

    chosen_analysis_id = st.selectbox(
        "Choose analysis for report",
        analysis_ids,
        format_func=lambda aid: f"{st.session_state.analysis_results[aid]['timestamp']} · {st.session_state.analysis_results[aid]['file_name']}",
    )
    result = st.session_state.analysis_results[chosen_analysis_id]

    top_label = result["primary_class"].replace("_", " ")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Primary class</div><div class="metric-value">{top_label}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Top confidence</div><div class="metric-value">{result["top_conf"]*100:.1f}%</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Detected findings</div><div class="metric-value">{len(result["detected"])}</div></div>', unsafe_allow_html=True)

    ai_key = f"report_ai_{chosen_analysis_id}"
    if ai_key not in st.session_state.ai_explanations:
        st.info("Generate the AI-assisted clinical narrative to enable editing and PDF export.")
        if st.button("Generate AI Explanation", key=f"gen_ai_{chosen_analysis_id}"):
            placeholder = st.empty()
            full_text = ""
            with st.spinner("Generating clinical explanation"):
                for chunk in stream_ai_explanation(result["detected"], result["patient_snapshot"], result["threshold"]):
                    full_text += chunk
                    placeholder.markdown(
                        f"""
                        <div style="border:1px solid rgba(148,163,184,0.35);border-radius:16px;padding:14px 16px;
                        background:rgba(255,255,255,0.72);backdrop-filter:blur(10px);min-height:320px;white-space:pre-wrap;
                        font-size:0.95rem;line-height:1.6;">{full_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')}</div>
                        """,
                        unsafe_allow_html=True,
                    )
            st.session_state.ai_explanations[ai_key] = full_text
            st.session_state.report_edits[chosen_analysis_id] = full_text
            st.rerun()
    else:
        if chosen_analysis_id not in st.session_state.report_edits:
            st.session_state.report_edits[chosen_analysis_id] = st.session_state.ai_explanations[ai_key]

    edited_text = st.text_area(
        "Editable clinical report",
        value=st.session_state.report_edits.get(chosen_analysis_id, ""),
        height=360,
        key=f"editor_{chosen_analysis_id}",
    )
    st.session_state.report_edits[chosen_analysis_id] = edited_text

    with st.expander("Preview images used in report"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.image(result["img_np"], caption="Original X-ray", use_container_width=True)
        with col_b:
            st.image(result["cam_img"], caption="Grad-CAM overlay", use_container_width=True)

    if st.button("Build PDF Report", key=f"build_pdf_{chosen_analysis_id}", use_container_width=True):
        with st.spinner("Building PDF report"):
            pdf_buf = build_pdf(
                result["patient_snapshot"],
                result["detected"],
                result["preds"],
                result["img_np"],
                result["cam_img"],
                result["threshold"],
                edited_ai_text=edited_text,
            )
            st.session_state[f"pdf_{chosen_analysis_id}"] = pdf_buf.getvalue()
            st.success("PDF built successfully.")

    stored_pdf = st.session_state.get(f"pdf_{chosen_analysis_id}")
    if stored_pdf:
        st.download_button(
            "Download Final PDF Report",
            data=stored_pdf,
            file_name=f"ThoraxAI_Report_{result['file_name']}.pdf",
            mime="application/pdf",
            key=f"download_pdf_{chosen_analysis_id}",
            use_container_width=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    if len(analysis_ids) >= 2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Trend Tracking</div>', unsafe_allow_html=True)
        trend_rows = []
        for aid in analysis_ids:
            item = st.session_state.analysis_results[aid]
            trend_rows.append({
                "Timestamp": item["timestamp"],
                "Primary Class": item["primary_class"].replace("_", " "),
                "Top Confidence": item["top_conf"],
                "Findings": len(item["detected"]),
            })
        trend_df = pd.DataFrame(trend_rows)
        st.line_chart(trend_df.set_index("Timestamp")[["Top Confidence"]], use_container_width=True)
        st.dataframe(trend_df.style.format({"Top Confidence": "{:.3f}"}), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

def render_patient_sessions_step():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Patient Sessions</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Review all patients currently loaded in this session, inspect their analyses, and compare scan history without persistence beyond refresh.</div>', unsafe_allow_html=True)

    if not st.session_state.patients:
        st.info("No patients have been added in this session yet.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    patient_ids = list(st.session_state.patients.keys())
    selected = st.selectbox(
        "Patient registry",
        patient_ids,
        index=patient_ids.index(st.session_state.selected_patient_for_dashboard) if st.session_state.selected_patient_for_dashboard in patient_ids else 0,
        format_func=lambda x: f"{x} — {st.session_state.patients[x]['full_name']}",
    )
    st.session_state.selected_patient_for_dashboard = selected
    st.session_state.active_patient_id = selected

    patient = st.session_state.patients[selected]
    st.markdown(f"### {patient['full_name']} · {patient['patient_id']}")
    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.write(f"**Age:** {patient['age']}")
        st.write(f"**Gender:** {patient['gender']}")
        st.write(f"**Smoking status:** {patient['smoker']}")
    with info_col2:
        st.write(f"**Pack-years:** {patient['pack_years']}")
        st.write(f"**Occupation:** {patient['occupation']}")
        st.write(f"**Thoracic surgery:** {'Yes' if patient['thoracic_surgery'] else 'No'}")
    with info_col3:
        st.write(f"**SpO₂:** {patient['spo2']}")
        st.write(f"**Resp. rate:** {patient['resp_rate']}")
        st.write(f"**Symptoms:** {', '.join(patient['symptoms']) or 'None'}")

    analysis_ids = patient.get("analyses", [])
    if not analysis_ids:
        st.info("No analyses saved for this patient yet.")
    else:
        rows = []
        for aid in analysis_ids:
            item = st.session_state.analysis_results[aid]
            rows.append({
                "Timestamp": item["timestamp"],
                "File": item["file_name"],
                "Primary Class": item["primary_class"].replace("_", " "),
                "Top Confidence": item["top_conf"],
                "Findings": len(item["detected"]),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df.style.format({"Top Confidence": "{:.3f}"}), use_container_width=True)
        st.line_chart(df.set_index("Timestamp")[["Top Confidence"]], use_container_width=True)

        inspect_id = st.selectbox(
            "Inspect saved analysis",
            analysis_ids,
            format_func=lambda aid: f"{st.session_state.analysis_results[aid]['timestamp']} · {st.session_state.analysis_results[aid]['file_name']}",
            key="inspect_saved_analysis",
        )
        item = st.session_state.analysis_results[inspect_id]
        col1, col2 = st.columns(2)
        with col1:
            st.image(item["img_np"], caption="Original X-ray", use_container_width=True)
        with col2:
            st.image(item["cam_img"], caption="Grad-CAM overlay", use_container_width=True)
        if item["detected"]:
            chip_html = "".join([f'<span class="pill {severity_color_key(label)}">{label.replace("_", " ")} · {prob*100:.0f}%</span>' for label, prob in item["detected"]])
            st.markdown(chip_html, unsafe_allow_html=True)

    if st.session_state.history:
        with st.expander("Session-wide analysis log"):
            hist_df = pd.DataFrame(st.session_state.history)
            st.dataframe(hist_df, use_container_width=True)
            st.download_button(
                "Export session log as CSV",
                data=hist_df.to_csv(index=False).encode("utf-8"),
                file_name="thoraxAI_session_log.csv",
                mime="text/csv",
                use_container_width=True,
            )
    if st.button("Clear Entire Session", type="secondary", use_container_width=True):
        for key in [
            "history", "ai_explanations", "patients", "active_patient_id",
            "analysis_results", "report_edits", "selected_patient_for_dashboard"
        ]:
            st.session_state[key] = {} if key in ["ai_explanations", "patients", "analysis_results", "report_edits"] else [] if key == "history" else None
        st.session_state.current_step = 1
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────
header()
stepper()

if st.session_state.current_step == 1:
    render_intake_step()
elif st.session_state.current_step == 2:
    render_analysis_step()
elif st.session_state.current_step == 3:
    render_report_step()
else:
    render_patient_sessions_step()

st.markdown(
    '<div class="footer-note">⚠ For research, education, and decision-support use only. This system does not replace a qualified radiologist or clinician. All outputs should be clinically reviewed before action.</div>',
    unsafe_allow_html=True,
)
