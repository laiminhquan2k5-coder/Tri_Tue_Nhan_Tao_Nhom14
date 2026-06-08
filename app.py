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
# CSS TUỲ BIẾN — GIAO DIỆN HIỆN ĐẠI
# ═══════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Gradient Header ── */
.header-gradient {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem 2rem 1.5rem 2rem;
    border-radius: 0 0 1.5rem 1.5rem;
    margin: -1rem -1rem 1.5rem -1rem;
    color: white;
    text-align: center;
}
.header-gradient h1 {
    color: white !important;
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    margin-bottom: 0.3rem !important;
    letter-spacing: -0.5px;
}
.header-gradient p {
    color: rgba(255,255,255,0.9) !important;
    font-size: 1.1rem !important;
    margin-top: 0 !important;
}

/* ── Card chung ── */
.card {
    background: #ffffff;
    border-radius: 1rem;
    padding: 1.5rem;
    box-shadow: 0 2px 12px rgba(102, 126, 234, 0.10);
    border: 1px solid #f0f0f5;
    margin-bottom: 1rem;
}
.card h3 {
    margin-top: 0 !important;
    color: #4a4a6a;
    font-weight: 700;
}

/* ── Card kết quả chính (gradient) ── */
.result-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 1rem;
    padding: 2rem;
    color: white;
    text-align: center;
    box-shadow: 0 4px 20px rgba(102, 126, 234, 0.30);
    margin-bottom: 1rem;
}
.result-card h2 {
    color: white !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    margin-bottom: 0.2rem !important;
}
.result-card .confidence {
    font-size: 1.2rem;
    opacity: 0.9;
}
.result-card .emoji-big {
    font-size: 3.5rem;
    margin-bottom: 0.3rem;
}

/* ── Stat card ── */
.stat-card {
    background: #f8f9ff;
    border-radius: 0.8rem;
    padding: 1rem;
    text-align: center;
    border: 1px solid #e8e8f0;
}
.stat-card .stat-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #667eea;
}
.stat-card .stat-label {
    font-size: 0.85rem;
    color: #888;
    margin-top: 0.2rem;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8f9ff 0%, #eeeef8 100%);
}
section[data-testid="stSidebar"] .sidebar-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #4a4a6a;
    margin-bottom: 0.5rem;
}

/* ── Nút phân tích ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 0.7rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.6rem 2rem !important;
    width: 100%;
    transition: transform 0.15s, box-shadow 0.15s;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}

/* ── Ẩn streamlit branding ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ── Dataframe đẹp ── */
.dataframe { border-radius: 0.5rem !important; }

/* ── Tag nhãn cảm xúc ── */
.sentiment-tag {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 2rem;
    font-weight: 600;
    font-size: 0.85rem;
    margin: 0.15rem;
}
.tag-neutral { background: #e0e0e0; color: #555; }
.tag-positive { background: #c8e6c9; color: #2e7d32; }
.tag-very-positive { background: #bbdefb; color: #1565c0; }

/* ── History item ── */
.history-item {
    background: #f8f9ff;
    border-radius: 0.5rem;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.4rem;
    border-left: 3px solid #667eea;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# HẰNG SỐ & CẤU HÌNH
# ═══════════════════════════════════════════════════════════════════════

SENTIMENT_MAP = {0: "Neutral", 1: "Positive", 2: "Very Positive"}
SENTIMENT_EMOJI = {0: "😐", 1: "😊", 2: "🤩"}
SENTIMENT_COLOR = {0: "#9E9E9E", 1: "#4CAF50", 2: "#2196F3"}
SENTIMENT_VN = {0: "Trung tính", 1: "Tích cực", 2: "Rất tích cực"}

ASPECT_COLS = ["Price", "Shipping", "Outlook", "Quality", "Size", "Shop_Service", "General", "Others"]

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
    # 1. Lowercase
    text = text.lower()
    # 2. Remove punctuation (giữ lại , . ! ? ; : - ( ))
    text = text.translate(PUNCT_TABLE)
    # 3. Remove URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # 4. Remove emoji (giữ lại , . ! ? ; : - ( ) / %)
    text = re.sub(r"[^\w\s,.!?;:\-()/%]|_", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    # 5. Chuẩn hóa teencode
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
    model_path = os.path.join(model_dir, "naive_bayes_countvec.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "naive_bayes_tfidf.pkl")
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline


# ═══════════════════════════════════════════════════════════════════════
# HÀM TIỆN ÍCH
# ═══════════════════════════════════════════════════════════════════════

def predict_sentiment(pipeline, text):
    """Dự đoán cảm xúc cho 1 văn bản."""
    cleaned = preprocess_text(text)
    pred = pipeline.predict([cleaned])[0]
    proba = pipeline.predict_proba([cleaned])[0]
    return pred, proba, cleaned


def predict_batch(pipeline, texts):
    """Dự đoán cảm xúc cho nhiều văn bản."""
    cleaned = [preprocess_text(t) for t in texts]
    preds = pipeline.predict(cleaned)
    probas = pipeline.predict_proba(cleaned)
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


# ═══════════════════════════════════════════════════════════════════════
# KHỞI TẠO SESSION STATE
# ═══════════════════════════════════════════════════════════════════════

if "history" not in st.session_state:
    st.session_state.history = []

# ═══════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="header-gradient">
    <h1>👟 ShoeSenti AI</h1>
    <p>Phân tích cảm xúc đánh giá sản phẩm giày dép bằng mô hình ViSoBERT + Naive Bayes</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🧠 Về mô hình")
    st.markdown("""
    **ShoeSenti AI** sử dụng kết hợp:
    - **ViSoBERT** (uitnlp/visobert) cho tokenization tiếng Việt
    - **Naive Bayes** (CountVectorizer) cho phân loại cảm xúc

    Pipeline tiền xử lý 8 bước:
    1. Lowercase
    2. Remove punctuation
    3. Remove URL
    4. Remove emoji
    5. Chuẩn hóa teencode
    6. Tokenization ViSoBERT
    7. Padding sequence
    8. Attention Mask
    """)

    st.markdown("---")
    st.markdown("### 🏷️ Nhãn cảm xúc")
    for label_id, (name, vn, emoji, color) in {
        0: ("Neutral", "Trung tính", "😐", "#9E9E9E"),
        1: ("Positive", "Tích cực", "😊", "#4CAF50"),
        2: ("Very Positive", "Rất tích cực", "🤩", "#2196F3"),
    }.items():
        st.markdown(f'<span style="background:{color};color:white;padding:0.2rem 0.8rem;border-radius:1rem;font-weight:600;font-size:0.9rem;">{emoji} {name} ({vn})</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📂 File hỗ trợ")
    st.markdown("- CSV (.csv)")
    st.markdown("- Excel (.xlsx)")
    st.markdown("- Cột văn bản: **Review**")

    st.markdown("---")
    st.markdown("### 📊 8 nhãn khía cạnh")
    for col in ASPECT_COLS:
        st.markdown(f"- {col}")

# ═══════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════════════════

pipeline = load_model()

if pipeline is None:
    st.error("⚠️ Không tìm thấy mô hình! Vui lòng huấn luyện mô hình trước.")
    st.info("Chạy `Naive_Bayes_model.py` để huấn luyện và lưu mô hình.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════
# CHẾ ĐỘ PHÂN TÍCH
# ═══════════════════════════════════════════════════════════════════════

mode = st.radio(
    "🔧 Chế độ phân tích",
    ["📝 Phân tích 1 đánh giá", "📁 Phân tích hàng loạt từ file"],
    horizontal=True,
)

# ═══════════════════════════════════════════════════════════════════════
# CHẾ ĐỘ 1: PHÂN TÍCH 1 ĐÁNH GIÁ
# ═══════════════════════════════════════════════════════════════════════

if mode == "📝 Phân tích 1 đánh giá":
    col_input, col_stats = st.columns([3, 2])

    with col_input:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### ✍️ Nhập đánh giá giày dép")
        user_input = st.text_area(
            "",
            placeholder="Ví dụ: Giày đẹp, đi êm lắm, giao hàng nhanh, đáng tiền!",
            height=150,
            label_visibility="collapsed",
        )
        analyze_btn = st.button("🔍 Phân tích cảm xúc", key="single", type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_stats:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📊 Thống kê nhanh")
        total = len(st.session_state.history)
        if total > 0:
            labels = [h["label"] for h in st.session_state.history]
            most_common = max(set(labels), key=labels.count)
            st.markdown(f"""
            <div class="stat-card" style="margin-bottom:0.5rem">
                <div class="stat-value">{total}</div>
                <div class="stat-label">Tổng phân tích</div>
            </div>
            <div class="stat-card" style="margin-bottom:0.5rem">
                <div class="stat-value">{SENTIMENT_EMOJI[most_common]}</div>
                <div class="stat-label">Cảm xúc phổ biến nhất</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="stat-card" style="margin-bottom:0.5rem">
                <div class="stat-value">0</div>
                <div class="stat-label">Tổng phân tích</div>
            </div>
            <div class="stat-card" style="margin-bottom:0.5rem">
                <div class="stat-value">—</div>
                <div class="stat-label">Chưa có dữ liệu</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

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
        st.markdown("---")
        res_col1, res_col2 = st.columns([2, 3])

        with res_col1:
            st.markdown(f"""
            <div class="result-card">
                <div class="emoji-big">{emoji}</div>
                <h2>{label}</h2>
                <p style="font-size:1rem;opacity:0.85;">{label_vn}</p>
                <div class="confidence">Độ tin cậy: <b>{confidence:.1f}%</b></div>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📈 Xác suất từng lớp")
            for i, (lbl, vn, emj) in {
                0: ("Neutral", "Trung tính", "😐"),
                1: ("Positive", "Tích cực", "😊"),
                2: ("Very Positive", "Rất tích cực", "🤩"),
            }.items():
                pct = proba[i] * 100
                bar_color = SENTIMENT_COLOR[i]
                st.markdown(
                    f'<div style="margin-bottom:0.6rem">'
                    f'<span style="font-weight:600">{emj} {lbl}</span>'
                    f'<span style="color:#888;font-size:0.85rem"> ({vn})</span><br>'
                    f'<div style="background:#e8e8f0;border-radius:0.5rem;overflow:hidden;height:1.2rem">'
                    f'<div style="background:{bar_color};width:{pct:.1f}%;height:100%;border-radius:0.5rem"></div>'
                    f'</div>'
                    f'<span style="font-size:0.85rem;color:#555">{pct:.1f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

        # Hiển thị text đã tiền xử lý
        with st.expander("🔧 Xem văn bản đã tiền xử lý"):
            st.code(cleaned, language="text")

    elif analyze_btn and not user_input.strip():
        st.warning("⚠️ Vui lòng nhập đánh giá trước khi phân tích!")

# ═══════════════════════════════════════════════════════════════════════
# CHẾ ĐỘ 2: PHÂN TÍCH HÀNG LOẠT TỪ FILE
# ═══════════════════════════════════════════════════════════════════════

else:
    col_upload, col_info = st.columns([3, 2])

    with col_upload:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📁 Tải lên file dữ liệu")
        uploaded_file = st.file_uploader(
            "Chọn file CSV hoặc Excel",
            type=["csv", "xlsx"],
            help="File phải chứa cột văn bản đánh giá",
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
                else:
                    df = pd.read_excel(uploaded_file)
                st.success(f"✅ Đọc thành công: **{len(df):,} dòng**, {len(df.columns)} cột")

                # Chọn cột text
                text_col = st.selectbox(
                    "📝 Chọn cột văn bản",
                    options=df.columns.tolist(),
                    index=df.columns.tolist().index("Review") if "Review" in df.columns else 0,
                )

                # Chọn cột khía cạnh (nếu có)
                aspect_cols = [c for c in ASPECT_COLS if c in df.columns]
                if aspect_cols:
                    st.info(f"📊 Phát hiện {len(aspect_cols)} cột khía cạnh: {', '.join(aspect_cols)}")

                analyze_batch_btn = st.button("🔍 Phân tích hàng loạt", key="batch", type="primary")

            except Exception as e:
                st.error(f"❌ Lỗi đọc file: {e}")
                df = None
                analyze_batch_btn = False
        else:
            df = None
            analyze_batch_btn = False
            st.info("👆 Tải lên file CSV hoặc Excel để bắt đầu")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_info:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📋 Hướng dẫn")
        st.markdown("""
        1. Tải lên file CSV hoặc Excel
        2. Chọn cột chứa văn bản đánh giá
        3. Bấm **Phân tích hàng loạt**
        4. Xem kết quả và tải xuống
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # Phân tích batch
    if analyze_batch_btn and df is not None and text_col:
        with st.spinner("🔄 Đang phân tích..."):
            texts = df[text_col].astype(str).tolist()
            preds, probas, cleaned = predict_batch(pipeline, texts)

            # Tạo DataFrame kết quả
            df_result = df.copy()
            df_result["Review_Cleaned"] = cleaned
            df_result["Sentiment"] = preds
            df_result["Sentiment_Label"] = [SENTIMENT_MAP[p] for p in preds]
            df_result["Confidence"] = [f"{probas[i][p]*100:.1f}%" for i, p in enumerate(preds)]

            for i, label in enumerate([0, 1, 2]):
                df_result[f"Prob_{SENTIMENT_MAP[label]}"] = [f"{probas[j][i]*100:.1f}%" for j in range(len(preds))]

        # Thống kê
        st.markdown("---")
        st.markdown("## 📊 Kết quả phân tích hàng loạt")

        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        total = len(preds)
        dist = pd.Series(preds).value_counts()

        with stat_col1:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{total:,}</div><div class="stat-label">Tổng đánh giá</div></div>', unsafe_allow_html=True)
        with stat_col2:
            pos_count = dist.get(1, 0) + dist.get(2, 0)
            st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:#4CAF50">{pos_count:,}</div><div class="stat-label">Tích cực</div></div>', unsafe_allow_html=True)
        with stat_col3:
            neu_count = dist.get(0, 0)
            st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:#9E9E9E">{neu_count:,}</div><div class="stat-label">Trung tính</div></div>', unsafe_allow_html=True)
        with stat_col4:
            avg_conf = np.mean([probas[i][preds[i]] for i in range(len(preds))]) * 100
            st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:#764ba2">{avg_conf:.1f}%</div><div class="stat-label">Độ tin cậy TB</div></div>', unsafe_allow_html=True)

        # Biểu đồ phân bố
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📊 Phân bố cảm xúc")
            dist_df = pd.DataFrame({
                "Cảm xúc": [SENTIMENT_EMOJI.get(k, "") + " " + SENTIMENT_MAP.get(k, str(k)) for k in dist.index],
                "Số lượng": dist.values,
            })
            st.bar_chart(dist_df.set_index("Cảm xúc"), height=300)
            st.markdown('</div>', unsafe_allow_html=True)

        with chart_col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🥧 Tỷ lệ cảm xúc")
            pie_df = pd.DataFrame({
                "Cảm xúc": [SENTIMENT_MAP.get(k, str(k)) for k in dist.index],
                "Tỷ lệ": dist.values / total * 100,
            })
            st.bar_chart(pie_df.set_index("Cảm xúc"), height=300)
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
# LỊCH SỬ PHÂN TÍCH
# ═══════════════════════════════════════════════════════════════════════

if st.session_state.history:
    st.markdown("---")
    st.markdown("### 🕐 Lịch sử phân tích gần đây")
    for i, item in enumerate(st.session_state.history[:10]):
        label = item["label"]
        emoji = SENTIMENT_EMOJI[label]
        label_name = SENTIMENT_MAP[label]
        conf = item["confidence"]
        text_preview = item["text"][:60] + ("..." if len(item["text"]) > 60 else "")
        st.markdown(
            f'<div class="history-item">'
            f'{emoji} <b>{label_name}</b> ({conf:.1f}%) — '
            f'<span style="color:#666">{text_preview}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#999;font-size:0.85rem;">'
    'ShoeSenti AI — Ứng dụng ViSoBERT phân tích cảm xúc đánh giá giày dép | '
    'Đồ án Trí tuệ nhân tạo — Nhóm 14</p>',
    unsafe_allow_html=True,
)