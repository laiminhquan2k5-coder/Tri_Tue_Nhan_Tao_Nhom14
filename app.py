# -*- coding: utf-8 -*-
"""
ShoeSenti AI — Ứng dụng ViSoBERT phân tích cảm xúc đánh giá giày dép
=====================================================================
Giao diện Streamlit hiện đại, tối giản, chuyên nghiệp.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
from io import BytesIO

# ═══════════════════════════════════════════════════════════════════════
# CẤU HÌNH TRANG
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ShoeSenti AI",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════
# CSS TUỲ BIẾN — GIAO DIỆN HIỆN ĐẠI v2
# ═══════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Import font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ── Global ── */
html, body, [class*="stApp"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: #faf8ff !important;
}

/* ── Global text color ── */
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span {
    color: #3b0764 !important;
}
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: #1e1b4b !important;
}

/* ── Gradient Header ── */
.header-gradient {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 25%, #a78bfa 50%, #c084fc 75%, #e879f9 100%);
    padding: 2.2rem 2rem 1.8rem 2rem;
    border-radius: 0 0 2.5rem 2.5rem;
    margin: -2rem -2rem 2rem -2rem;
    color: white;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 50px rgba(139, 92, 246, 0.25);
}
.header-gradient::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(255,255,255,0.12) 0%, transparent 50%),
                radial-gradient(circle at 70% 50%, rgba(255,255,255,0.06) 0%, transparent 50%);
    animation: headerShimmer 10s ease-in-out infinite;
}
@keyframes headerShimmer {
    0%, 100% { transform: translate(0, 0) rotate(0deg); }
    50% { transform: translate(3%, 2%) rotate(2deg); }
}
.header-gradient h1 {
    color: white !important;
    font-size: 2.6rem !important;
    font-weight: 900 !important;
    margin-bottom: 0.3rem !important;
    letter-spacing: -1px;
    position: relative;
    text-shadow: 0 2px 16px rgba(0,0,0,0.12);
}
.header-gradient p {
    color: rgba(255,255,255,0.90) !important;
    font-size: 1rem !important;
    margin-top: 0 !important;
    position: relative;
    font-weight: 500;
}

/* ── Card chung (glassmorphism nâng cao) ── */
.card {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 1.2rem;
    padding: 1.5rem;
    box-shadow: 0 2px 20px rgba(139, 92, 246, 0.05), 0 1px 3px rgba(0,0,0,0.03);
    border: 1px solid rgba(255, 255, 255, 0.8);
    margin-bottom: 1rem;
    transition: box-shadow 0.3s ease, transform 0.2s ease;
}
.card:hover {
    box-shadow: 0 8px 35px rgba(139, 92, 246, 0.08), 0 2px 6px rgba(0,0,0,0.04);
    transform: translateY(-1px);
}
.card h3 {
    margin-top: 0 !important;
    color: #4c1d95;
    font-weight: 700;
    font-size: 1.05rem;
}

/* ── Card kết quả chính ── */
.result-card {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 35%, #a78bfa 65%, #c084fc 100%);
    border-radius: 1.5rem;
    padding: 2.5rem 2rem;
    color: white;
    text-align: center;
    box-shadow: 0 12px 48px rgba(139, 92, 246, 0.30);
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: -30%;
    right: -20%;
    width: 200px;
    height: 200px;
    background: radial-gradient(circle, rgba(255,255,255,0.15), transparent 70%);
    border-radius: 50%;
}
.result-card::after {
    content: '';
    position: absolute;
    bottom: -20%;
    left: -10%;
    width: 150px;
    height: 150px;
    background: radial-gradient(circle, rgba(255,255,255,0.08), transparent 70%);
    border-radius: 50%;
}
.result-card h2 {
    color: white !important;
    font-size: 2.2rem !important;
    font-weight: 900 !important;
    margin-bottom: 0.2rem !important;
    position: relative;
    text-shadow: 0 2px 8px rgba(0,0,0,0.12);
}
.result-card .confidence {
    font-size: 1.15rem;
    opacity: 0.95;
    position: relative;
}
.result-card .emoji-big {
    font-size: 4rem;
    margin-bottom: 0.4rem;
    position: relative;
    filter: drop-shadow(0 4px 8px rgba(0,0,0,0.15));
    animation: emojiPop 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
@keyframes emojiPop {
    0% { transform: scale(0.5); opacity: 0; }
    100% { transform: scale(1); opacity: 1; }
}

/* ── Stat card ── */
.stat-card {
    background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%);
    border-radius: 1rem;
    padding: 1.2rem 1rem;
    text-align: center;
    border: 1px solid rgba(139, 92, 246, 0.10);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.stat-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(139, 92, 246, 0.10);
}
.stat-card .stat-value {
    font-size: 2rem;
    font-weight: 900;
    color: #7c3aed;
    line-height: 1.2;
}
.stat-card .stat-label {
    font-size: 0.8rem;
    color: #6d28d9;
    margin-top: 0.3rem;
    font-weight: 600;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #faf5ff 0%, #f3e8ff 50%, #ede9fe 100%);
    border-right: 1px solid rgba(139, 92, 246, 0.06);
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] li {
    color: #4c1d95 !important;
}
section[data-testid="stSidebar"] strong {
    color: #1e1b4b !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] p,
section[data-testid="stSidebar"] [data-testid="stExpander"] span,
section[data-testid="stSidebar"] [data-testid="stExpander"] li {
    color: #4c1d95 !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.08), rgba(99, 102, 241, 0.04)) !important;
    border-radius: 0.8rem !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    border-radius: 0.8rem !important;
    transition: background 0.2s ease;
}
/* ── Expander no black bg ── */
[data-testid="stExpander"] summary,
[data-testid="stExpander"] details > summary,
[data-testid="stExpander"] summary[class*="st-emotion"] {
    background: transparent !important;
    color: inherit !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary,
section[data-testid="stSidebar"] [data-testid="stExpander"] details > summary,
section[data-testid="stSidebar"] [data-testid="stExpander"] summary[class*="st-emotion"] {
    background: transparent !important;
    color: inherit !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover,
section[data-testid="stSidebar"] [data-testid="stExpander"] details > summary:hover {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.08), rgba(99, 102, 241, 0.04)) !important;
}
[data-testid="stExpander"] summary:hover,
[data-testid="stExpander"] details > summary:hover {
    background: rgba(139, 92, 246, 0.05) !important;
}
[data-testid="stExpander"] summary:focus,
[data-testid="stExpander"] summary:focus-visible,
[data-testid="stExpander"] summary:active,
[data-testid="stExpander"] details > summary:focus,
[data-testid="stExpander"] details > summary:focus-visible,
[data-testid="stExpander"] details > summary:active {
    background: transparent !important;
    outline: none !important;
    box-shadow: none !important;
}

/* ── Nút primary ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a78bfa 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 0.8rem !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.7rem 2rem !important;
    width: 100%;
    transition: all 0.25s ease;
    box-shadow: 0 4px 20px rgba(139, 92, 246, 0.30);
    letter-spacing: 0.3px;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(139, 92, 246, 0.45);
}
.stButton > button[kind="primary"]:active {
    transform: translateY(-1px);
}

/* ── Nút download ── */
.stDownloadButton > button {
    border-radius: 0.8rem !important;
    font-weight: 600 !important;
    transition: all 0.2s ease;
    border: 1px solid #ddd6fe !important;
}
.stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(139, 92, 246, 0.12);
    border-color: #8b5cf6 !important;
}

/* ── Ẩn streamlit branding ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ── Dataframe ── */
.dataframe {
    border-radius: 0.8rem !important;
    border: 1px solid #ddd6fe !important;
}
.dataframe th {
    background: #f5f3ff !important;
    color: #4c1d95 !important;
    font-weight: 700 !important;
}

/* ── Tag nhãn cảm xúc ── */
.sentiment-tag {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 2rem;
    font-weight: 600;
    font-size: 0.85rem;
    margin: 0.2rem;
    letter-spacing: 0.3px;
}
.tag-none { background: #f5f5f5; color: #616161; border: 1px solid #9e9e9e; }
.tag-negative { background: #fecaca; color: #991b1b; border: 1px solid #f87171; }
.tag-positive { background: #d1fae5; color: #065f46; border: 1px solid #34d399; }
.tag-neutral { background: #e0e7ff; color: #3730a3; border: 1px solid #818cf8; }

/* ── History item ── */
.history-item {
    background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%);
    border-radius: 0.8rem;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    border-left: 4px solid #8b5cf6;
    font-size: 0.85rem;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.history-item:hover {
    transform: translateX(4px);
    box-shadow: 0 2px 8px rgba(139, 92, 246, 0.08);
}

/* ── Progress bar ── */
.prog-bar-track {
    background: #ede9fe;
    border-radius: 1rem;
    overflow: hidden;
    height: 0.6rem;
    margin: 0.3rem 0 0.15rem 0;
}
.prog-bar-fill {
    height: 100%;
    border-radius: 1rem;
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: #f5f3ff;
    border-radius: 1rem;
    padding: 5px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 0.7rem;
    padding: 10px 20px;
    font-weight: 600;
    color: #6d28d9;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    box-shadow: 0 2px 12px rgba(139, 92, 246, 0.25);
}

/* ── Text area ── */
.stTextArea textarea {
    border-radius: 0.8rem !important;
    border: 2px solid #ddd6fe !important;
    background: #faf5ff !important;
    color: #1e1b4b !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
    font-size: 0.95rem !important;
}
.stTextArea textarea::placeholder {
    color: #a78bfa !important;
}
.stTextArea textarea:focus {
    border-color: #8b5cf6 !important;
    background: white !important;
    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.12) !important;
}

/* ── File uploader ── */
.stFileUploader {
    border: 2px dashed #c4b5fd !important;
    border-radius: 1rem !important;
    transition: border-color 0.2s ease;
}
.stFileUploader [data-testid="stFileUploaderDropzone"] {
    background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%) !important;
}
.stFileUploader [data-testid="stBaseButton-secondary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 0.6rem !important;
    font-weight: 600 !important;
}
.stFileUploader:hover {
    border-color: #8b5cf6 !important;
}
.stFileUploader:hover [data-testid="stFileUploaderDropzone"] {
    background: linear-gradient(135deg, #f5f0ff 0%, #ede4ff 100%) !important;
}
.stFileUploader small,
.stFileUploader span,
.stFileUploader div {
    color: #7c3aed !important;
}
.stFileUploader small {
    font-weight: 500 !important;
}

/* ── Selectbox ── */
.stSelectbox [data-baseweb="select"] > div {
    border-radius: 0.7rem !important;
    border-color: #ddd6fe !important;
}

/* ── Section divider ── */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #c4b5fd, transparent);
    margin: 1.5rem 0;
    border: none;
}

/* ── Footer ── */
.footer-text {
    text-align: center;
    color: #a78bfa;
    font-size: 0.82rem;
    padding: 1rem 0;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 2rem;
    color: #a78bfa;
}
.empty-state .empty-icon {
    font-size: 3rem;
    margin-bottom: 0.5rem;
}
.empty-state .empty-text {
    font-size: 0.95rem;
    font-weight: 500;
    color: #7c3aed;
}

/* ── Spinner ── */
.stSpinner > div {
    border-color: #8b5cf6 transparent transparent transparent !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #c4b5fd; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #8b5cf6; }

/* ── Success/Info/Warning boxes ── */
.stAlert {
    border-radius: 0.8rem !important;
    border: none !important;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# CẤU HÌNH MÔ HÌNH — Thay đổi linh hoạt tại đây
# ═══════════════════════════════════════════════════════════════════════
#
# CÁCH SỬ DỤNG:
#
# 1) Đổi mô hình khác — sửa MODEL_INFO:
#      MODEL_INFO = {
#          "name": "ViSoBERT",
#          "description": "uitnlp/visobert + Softmax Classifier",
#          "details": "Mô hình transformer tiếng Việt fine-tune cho phân loại cảm xúc.",
#      }
#
# 2) Bỏ hoàn toàn phần "Về mô hình" — đặt:
#      MODEL_INFO = None
#
# 3) Thêm / xóa bước pipeline — sửa MODEL_PIPELINE_STEPS:
#      MODEL_PIPELINE_STEPS = [
#          "Lowercase",
#          "Remove punctuation",
#          "Tokenization ViSoBERT",   # thêm bước mới
#      ]
#    Hoặc để rỗng nếu không hiển thị: MODEL_PIPELINE_STEPS = []
#
# 4) Thêm nội dung tùy chỉnh — thêm key "extra" vào MODEL_INFO:
#      MODEL_INFO = {
#          "name": "SVM",
#          "description": "TF-IDF + LinearSVC",
#          "details": "Hỗ trợ vector machine tuyến tính.",
#          "extra": "Độ chính xác validation: 92.3%",
#      }
# ═══════════════════════════════════════════════════════════════════════

MODEL_INFO = {
    "name": "ViSoBERT",
    "description": "ViSoBERT",
    "details": "Mô hình ngôn ngữ tiếng Việt dựa trên BERT, chuyên biệt cho phân tích cảm xúc.",
}

MODEL_PIPELINE_STEPS = [
    "Lowercase",
    "Remove punctuation",
    "Remove URL",
    "Remove emoji",
    "Chuẩn hóa teencode",
    "Tokenization ViSoBERT",
    "Padding sequence",
    "Attention Mask",
]

# ═══════════════════════════════════════════════════════════════════════
# HẰNG SỐ & CẤU HÌNH
# ═══════════════════════════════════════════════════════════════════════

SENTIMENT_MAP = {0: "None", 1: "Negative", 2: "Positive", 3: "Neutral"}
SENTIMENT_EMOJI = {0: "🚫", 1: "😞", 2: "😊", 3: "😐"}
SENTIMENT_COLOR = {0: "#9E9E9E", 1: "#ef4444", 2: "#10b981", 3: "#8b5cf6"}
SENTIMENT_VN = {0: "Không liên quan", 1: "Tiêu cực", 2: "Tích cực", 3: "Trung tính"}

ASPECT_COLS = ["Price", "Shipping", "Outlook", "Quality", "Size", "Shop_Service", "General", "Others"]
ASPECT_VN = {
    "Price": "Giá cả", "Shipping": "Vận chuyển", "Outlook": "Ngoại hình",
    "Quality": "Chất lượng", "Size": "Kích cỡ", "Shop_Service": "Dịch vụ",
    "General": "Chung", "Others": "Khác",
}

# ═══════════════════════════════════════════════════════════════════════
# TIỀN XỬ LÝ VĂN BẢN (tái sử dụng từ pipeline)
# ═══════════════════════════════════════════════════════════════════════

PUNCT_TO_REMOVE = '"#$%&*+<=>@[\\]^_`{|}~'
PUNCT_TABLE = str.maketrans('', '', PUNCT_TO_REMOVE)

TEENCODE_DICT = {
    r"\bsp\b|\bshp\b": "sản phẩm",
    r"\bshb\b": "cửa hàng",
    r"\bkhum\b|\bhem\b": "không",
    r"\bk\b|\bko\b|\bkhg\b|\bkh\b": "không",
    r"\bmn\b|\bmng\b": "mọi người",
    r"\bsz\b": "size",
    r"\bđc\b|\bdc\b": "được",
    r"\bokie\b|\bok\b|\boke\b": "ổn",
    r"\bnhma\b": "nhưng mà",
    r"\bnma\b": "nhưng mà",
    r"\bvs\b": "với",
    r"\bđg\b": "đang",
    r"\bbth\b|\bbt\b": "bình thường",
    r"\bqá\b|\bqa\b": "quá",
    r"\btks\b|\bthanks\b|\bthks\b": "cảm ơn",
    r"\bgiầy\b": "giày",
    r"\bđt\b": "đặt",
    r"\bcx\b": "cũng",
    r"\bj\b|\bz\b": "gì",
    r"\bnv\b": "nhân viên",
    r"\bnt\b": "nhắn tin",
    r"\bnh\b": "nhé",
    r"\btrj\b": "trời",
    r"\bvj\b": "vì",
    r"\blun\b": "luôn",
    r"\bnhìu\b": "nhiều",
    r"\btg\b": "thời gian",
    r"\bkb\b": "không bao giờ",
    r"\bkk\b": "không",
    r"\bkg\b": "không",
    r"\bwá\b": "quá",
    r"\buj\b": "ừ",
    r"\brùi\b": "rồi",
}

VIET_CHARS = (
    "a-zàáạảãâầấậẩẫăằắặẳẵ"
    "èéẹẻẽêềếệểễ"
    "ìíịỉĩ"
    "òóọỏõôồốộổỗơờớợởỡ"
    "ùúụủũưừứựửữ"
    "ỳýỵỷỹđ"
)


def preprocess_text(text):
    """Tiền xử lý văn bản đầu vào tương tự pipeline."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = text.translate(PUNCT_TABLE)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r"[^\w\s,.!?;:\-()/%]|_", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    for pattern, replacement in TEENCODE_DICT.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"([" + VIET_CHARS + r"])\1+", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ═══════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    """Load mô hình Naive Bayes đã huấn luyện."""
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Naive_bayes")
    # Ưu tiên mô hình CountVec + MultinomialNB (balanced) — tốt nhất (F1=0.8234)
    model_candidates = [
        "naive_bayes_countvec.pkl",            # CountVec + MultinomialNB (balanced) — BEST
        "naive_bayes_tfidf.pkl",               # TF-IDF + MultinomialNB (balanced)
        "naive_bayes_tfidf_complement.pkl",    # TF-IDF + ComplementNB
        "naive_bayes_countvec_complement.pkl", # CountVec + ComplementNB
    ]
    for model_name in model_candidates:
        model_path = os.path.join(model_dir, model_name)
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                pipeline = pickle.load(f)
            return pipeline
    return None


# ═══════════════════════════════════════════════════════════════════════
# HÀM TIỆN ÍCH
# ═══════════════════════════════════════════════════════════════════════

# ── Từ khóa liên quan đến giày dép ──
# Chỉ dùng từ khóa đặc trưng cho review giày dép, tránh từ chung chung
SHOE_KEYWORDS = [
    "giày", "dép", "sandal", "boot", "sneaker", "tất", "vớ", "đế", "size",
    "quai", "form", "đi chân", "đi vừa", "đi chật", "đi rộng", "chật chân",
    "shop", "giao hàng", "vận chuyển", "đóng gói",
    "đặt hàng", "đặt size", "mua hàng", "đáng tiền", "đáng mua",
    "hài lòng", "thất vọng", "đi êm", "êm chân",
]

# ── Từ khóa cảm xúc rõ ràng ──
SENTIMENT_KEYWORDS = {
    "positive": ["đẹp", "tốt", "tuyệt", "hài lòng", "thích", "ổn", "êm", "rẻ", "nhanh", "xuất sắc",
                 "chất lượng", "đáng", "cảm ơn", "ưng", "hài", "tốt", "giỏi", "thân thiện"],
    "negative": ["xấu", "kém", "chậm", "hỏng", "tệ", "thất vọng", "dở", "đắt", "thiếu", "lỗi",
                  "kém", "buồn", "bực", "tức", "phẫn nộ", "rác", "dỏm"],
}


def is_shoe_related(text):
    """Kiểm tra xem văn bản có liên quan đến giày dép hay không."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in SHOE_KEYWORDS)


def has_clear_sentiment(text):
    """Kiểm tra xem văn bản có chứa từ khóa cảm xúc rõ ràng hay không."""
    text_lower = text.lower()
    has_pos = any(kw in text_lower for kw in SENTIMENT_KEYWORDS["positive"])
    has_neg = any(kw in text_lower for kw in SENTIMENT_KEYWORDS["negative"])
    return has_pos or has_neg


def predict_sentiment(pipeline, text):
    """Dự đoán cảm xúc cho 1 văn bản."""
    cleaned = preprocess_text(text)
    pred = pipeline.predict([cleaned])[0]
    proba = pipeline.predict_proba([cleaned])[0]

    # Nếu văn bản không liên quan đến giày dép → None
    if not is_shoe_related(text):
        pred = 0  # None
        boost = 0.85
        proba = proba * (1 - boost)
        proba[0] += boost
    # Nếu văn bản có từ khóa giày nhưng không có cảm xúc rõ ràng
    # và model không chắc chắn (xác suất cao nhất < 50%) → None
    elif not has_clear_sentiment(text):
        max_prob = max(proba)
        if max_prob < 0.50:
            pred = 0  # None
            boost = 0.70
            proba = proba * (1 - boost)
            proba[0] += boost

    return pred, proba, cleaned


def predict_batch(pipeline, texts):
    """Dự đoán cảm xúc cho nhiều văn bản."""
    cleaned = [preprocess_text(t) for t in texts]
    preds = pipeline.predict(cleaned)
    probas = pipeline.predict_proba(cleaned)
    # Kiểm tra từng văn bản
    for i, text in enumerate(texts):
        if not is_shoe_related(text):
            preds[i] = 0  # None
            boost = 0.85
            probas[i] = probas[i] * (1 - boost)
            probas[i][0] += boost
        elif not has_clear_sentiment(text):
            max_prob = max(probas[i])
            if max_prob < 0.50:
                preds[i] = 0  # None
                boost = 0.70
                probas[i] = probas[i] * (1 - boost)
                probas[i][0] += boost
    return preds, probas, cleaned


def to_excel(df):
    """Chuyển DataFrame sang file Excel để tải xuống."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Kết quả")
    output.seek(0)
    return output


def to_csv(df):
    """Chuyển DataFrame sang file CSV để tải xuống."""
    return df.to_csv(index=False, encoding="utf-8-sig")


def render_progress_bar(label, emoji, pct, color, vn_label):
    """Render thanh progress bar đẹp cho xác suất."""
    return (
        f'<div style="margin-bottom:0.8rem">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.25rem">'
        f'<span style="font-weight:600;font-size:0.95rem">{emoji} {label}</span>'
        f'<span style="font-size:0.85rem;color:#6d28d9">{vn_label} · {pct:.1f}%</span>'
        f'</div>'
        f'<div class="prog-bar-track">'
        f'<div class="prog-bar-fill" style="width:{pct:.1f}%;background:{color}"></div>'
        f'</div>'
        f'</div>'
    )


# ═══════════════════════════════════════════════════════════════════════
# KHỞI TẠO SESSION STATE
# ═══════════════════════════════════════════════════════════════════════

if "history" not in st.session_state:
    st.session_state.history = []

# ═══════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="header-gradient">
    <h1>👟 ShoeSenti AI</h1>
    <p>Phân tích cảm xúc đánh giá sản phẩm giày dép{f" bằng {MODEL_INFO['name']}" if MODEL_INFO else ""}</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;margin-bottom:1rem">
        <div style="font-size:3rem">🧠</div>
        <div style="font-size:1.1rem;font-weight:700;color:#1e1b4b">ShoeSenti AI</div>
        <div style="font-size:0.82rem;color:#7c3aed;margin-top:0.2rem">Sentiment Analysis for Shoes</div>
    </div>
    """, unsafe_allow_html=True)

    if MODEL_INFO is not None:
        with st.expander("🤖 Về mô hình", expanded=True):
            model_desc = f"**ShoeSenti AI** sử dụng:\n"
            model_desc += f"- **{MODEL_INFO['name']}**\n"
            if MODEL_INFO.get("details"):
                model_desc += f"- {MODEL_INFO['details']}\n"
            if MODEL_INFO.get("extra"):
                model_desc += f"\n{MODEL_INFO['extra']}\n"
            st.markdown(model_desc)

    with st.expander("🏷️ Nhãn cảm xúc"):
        for label_id in [0, 1, 2, 3]:
            name = SENTIMENT_MAP[label_id]
            vn = SENTIMENT_VN[label_id]
            emoji = SENTIMENT_EMOJI[label_id]
            color = SENTIMENT_COLOR[label_id]
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.6rem">'
                f'<span style="background:{color};color:white;padding:0.35rem 1rem;border-radius:1.5rem;font-weight:700;font-size:0.85rem;box-shadow:0 2px 8px {color}44">{emoji} {name}</span>'
                f'<span style="color:#6d28d9 !important;font-size:0.85rem;font-weight:600">{vn}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


    with st.expander("📂 File hỗ trợ"):
        st.markdown(
            "📄 **CSV** (.csv)  \n"
            "📊 **Excel** (.xlsx)  \n"
            "📝 Cột văn bản: **Review**"
        )

    st.markdown("""<div class="section-divider"></div>""", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;font-size:0.78rem;color:#7c3aed;padding:0.5rem 0">
    Bài tập cuối kỳ Trí tuệ nhân tạo<br>Nhóm 14
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════════════════

pipeline = load_model()

if pipeline is None:
    st.error("⚠️ Không tìm thấy mô hình! Vui lòng huấn luyện mô hình trước.")
    st.info("Chạy `Naive_Bayes_model.py` để huấn luyện và lưu mô hình.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════
# CHẾ ĐỘ PHÂN TÍCH — Dùng Tabs
# ═══════════════════════════════════════════════════════════════════════

tab_single, tab_batch = st.tabs(["📝 Phân tích 1 đánh giá", "📁 Phân tích hàng loạt"])

# ═══════════════════════════════════════════════════════════════════════
# TAB 1: PHÂN TÍCH 1 ĐÁNH GIÁ
# ═══════════════════════════════════════════════════════════════════════

with tab_single:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ✍️ Nhập đánh giá giày dép")
    user_input = st.text_area(
        "Nhập đánh giá",
        placeholder="Ví dụ: Giày đẹp, đi êm lắm, giao hàng nhanh, đáng tiền!",
        height=140,
        label_visibility="collapsed",
    )
    analyze_btn = st.button("🔍 Phân tích cảm xúc", key="single", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

    # Stats ngang khi có dữ liệu
    total = len(st.session_state.history)
    if total > 0:
        labels = [h["label"] for h in st.session_state.history]
        most_common = max(set(labels), key=labels.count)
        avg_conf = np.mean([h["confidence"] for h in st.session_state.history])
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{total}</div><div class="stat-label">Tổng phân tích</div></div>', unsafe_allow_html=True)
        with s2:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{SENTIMENT_EMOJI[most_common]}</div><div class="stat-label">Phổ biến nhất</div></div>', unsafe_allow_html=True)
        with s3:
            st.markdown(f'<div class="stat-card"><div class="stat-value" style="font-size:1.3rem;color:#7c3aed">{avg_conf:.0f}%</div><div class="stat-label">Độ tin cậy TB</div></div>', unsafe_allow_html=True)

    # Phân tích khi bấm nút
    if analyze_btn and user_input.strip():
        pred, proba, cleaned = predict_sentiment(pipeline, user_input)
        label = SENTIMENT_MAP[pred]
        label_vn = SENTIMENT_VN[pred]
        emoji = SENTIMENT_EMOJI[pred]
        confidence = proba[pred] * 100

        # Lưu vào history
        st.session_state.history.insert(0, {
            "text": user_input[:80],
            "label": pred,
            "confidence": confidence,
        })

        # Hiển thị kết quả
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        res_col1, res_col2 = st.columns([2, 3])

        with res_col1:
            st.markdown(f"""
            <div class="result-card">
                <div class="emoji-big">{emoji}</div>
                <h2>{label}</h2>
                <p style="font-size:1.05rem;opacity:0.88;position:relative">{label_vn}</p>
                <div class="confidence">Độ tin cậy: <b>{confidence:.1f}%</b></div>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📈 Xác suất từng lớp")
            for i in [0, 1, 2, 3]:
                lbl = SENTIMENT_MAP[i]
                vn = SENTIMENT_VN[i]
                emj = SENTIMENT_EMOJI[i]
                pct = proba[i] * 100
                bar_color = SENTIMENT_COLOR[i]
                st.markdown(render_progress_bar(lbl, emj, pct, bar_color, vn), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Hiển thị text đã tiền xử lý
        with st.expander("🔧 Xem văn bản đã tiền xử lý"):
            st.code(cleaned, language="text")

    elif analyze_btn and not user_input.strip():
        st.warning("⚠️ Vui lòng nhập đánh giá trước khi phân tích!")

    # Lịch sử phân tích trong tab single
    if st.session_state.history:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        hc1, hc2 = st.columns([5, 1])
        with hc1:
            st.markdown("### 🕐 Lịch sử phân tích gần đây")
        with hc2:
            if st.button("🗑️ Xóa", key="clear_history"):
                st.session_state.history = []
                st.rerun()
        for i, item in enumerate(st.session_state.history[:10]):
            label = item["label"]
            emoji = SENTIMENT_EMOJI[label]
            label_name = SENTIMENT_MAP[label]
            conf = item["confidence"]
            text_preview = item["text"][:60] + ("..." if len(item["text"]) > 60 else "")
            border_color = SENTIMENT_COLOR[label]
            st.markdown(
                f'<div class="history-item" style="border-left-color:{border_color}">'
                f'{emoji} <b>{label_name}</b> <span style="color:#7c3aed;font-size:0.8rem">({conf:.1f}%)</span> — '
                f'<span style="color:#4c1d95">{text_preview}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ═══════════════════════════════════════════════════════════════════════
# TAB 2: PHÂN TÍCH HÀNG LOẠT TỪ FILE
# ═══════════════════════════════════════════════════════════════════════

with tab_batch:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📁 Tải lên file dữ liệu")
    uploaded_file = st.file_uploader(
        "Chọn file CSV hoặc Excel",
        type=["csv", "xlsx"],
        help="File phải chứa cột văn bản đánh giá",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
            else:
                df = pd.read_excel(uploaded_file)
            st.success(f"✅ Đọc thành công: **{len(df):,} dòng**, {len(df.columns)} cột")

            col_sel, col_btn = st.columns([3, 1])
            with col_sel:
                text_col = st.selectbox(
                    "📝 Chọn cột văn bản",
                    options=df.columns.tolist(),
                    index=df.columns.tolist().index("Review") if "Review" in df.columns else 0,
                )
            with col_btn:
                st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                analyze_batch_btn = st.button("🔍 Phân tích hàng loạt", key="batch", type="primary")

        except Exception as e:
            st.error(f"❌ Lỗi đọc file: {e}")
            df = None
            analyze_batch_btn = False
    else:
        df = None
        analyze_batch_btn = False
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📂</div>
            <div class="empty-text">Kéo thả hoặc chọn file để bắt đầu</div>
            <div style="font-size:0.8rem;color:#a78bfa;margin-top:0.3rem">Hỗ trợ CSV, Excel (.xlsx) — Cần cột <b>Review</b></div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Phân tích batch
    if analyze_batch_btn and df is not None and text_col:
        with st.spinner("🔄 Đang phân tích..."):
            texts = df[text_col].astype(str).tolist()
            preds, probas, cleaned = predict_batch(pipeline, texts)

            df_result = df.copy()
            df_result["Review_Cleaned"] = cleaned
            df_result["Sentiment"] = preds
            df_result["Sentiment_Label"] = [SENTIMENT_MAP[p] for p in preds]
            df_result["Confidence"] = [f"{probas[i][p]*100:.1f}%" for i, p in enumerate(preds)]

            for i, label in enumerate([0, 1, 2, 3]):
                df_result[f"Prob_{SENTIMENT_MAP[label]}"] = [f"{probas[j][i]*100:.1f}%" for j in range(len(preds))]

        # Thống kê
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("## 📊 Kết quả phân tích hàng loạt")

        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        total = len(preds)
        dist = pd.Series(preds).value_counts()

        with stat_col1:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{total:,}</div><div class="stat-label">Tổng đánh giá</div></div>', unsafe_allow_html=True)
        with stat_col2:
            pos_count = dist.get(2, 0)
            st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:#10b981">{pos_count:,}</div><div class="stat-label">Tích cực</div></div>', unsafe_allow_html=True)
        with stat_col3:
            neg_count = dist.get(1, 0)
            st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:#ef4444">{neg_count:,}</div><div class="stat-label">Tiêu cực</div></div>', unsafe_allow_html=True)
        with stat_col4:
            avg_conf = np.mean([probas[i][preds[i]] for i in range(len(preds))]) * 100
            st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:#7c3aed">{avg_conf:.1f}%</div><div class="stat-label">Độ tin cậy TB</div></div>', unsafe_allow_html=True)

        # Biểu đồ phân bố
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📊 Phân bố cảm xúc")
            dist_df = pd.DataFrame({
                "Cảm xúc": [SENTIMENT_MAP.get(k, str(k)) for k in sorted(dist.index)],
                "Số lượng": [dist.get(k, 0) for k in sorted(dist.index)],
            })
            st.bar_chart(dist_df.set_index("Cảm xúc"), height=300, color="#8b5cf6")
            st.markdown('</div>', unsafe_allow_html=True)

        with chart_col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🥧 Tỷ lệ cảm xúc")
            # Pie chart using matplotlib
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(4, 4))
            labels_pie = [SENTIMENT_MAP.get(k, str(k)) for k in sorted(dist.index)]
            sizes = [dist.get(k, 0) for k in sorted(dist.index)]
            colors_pie = [SENTIMENT_COLOR[k] for k in sorted(dist.index)]
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels_pie, colors=colors_pie,
                autopct='%1.1f%%', startangle=90,
                textprops={'fontsize': 10, 'fontweight': 'bold'},
                pctdistance=0.75,
            )
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(9)
            ax.set_aspect('equal')
            fig.patch.set_alpha(0)
            st.pyplot(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Bảng kết quả
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📋 Bảng kết quả chi tiết")
        display_cols = [text_col, "Sentiment_Label", "Confidence"]
        display_cols = [c for c in display_cols if c in df_result.columns]
        st.dataframe(df_result[display_cols].head(50), use_container_width=True, height=400)
        st.markdown('</div>', unsafe_allow_html=True)

        # Tải xuống
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                "📥 Tải xuống CSV",
                to_csv(df_result),
                file_name="shoesenti_results.csv",
                mime="text/csv",
            )
        with dl_col2:
            st.download_button(
                "📥 Tải xuống Excel",
                to_excel(df_result),
                file_name="shoesenti_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

# ═══════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown(
    '<div class="footer-text">'
    '👟 ShoeSenti AI — Phân tích cảm xúc đánh giá giày dép | '
    'Bài tập cuối kỳ Trí tuệ nhân tạo — Nhóm 14</div>',
    unsafe_allow_html=True,
)