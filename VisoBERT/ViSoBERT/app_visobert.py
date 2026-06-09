# -*- coding: utf-8 -*-
"""
ShoeSenti AI — Ứng dụng ViSoBERT phân tích cảm xúc đánh giá giày dép
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import re
from io import BytesIO
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

# Import mô hình ViSoBERT từ file của bạn
from Model import ViSoBERTMultiAspect, ASPECTS, MODEL_NAME, MAX_LEN, NUM_CLASSES

# ═══════════════════════════════════════════════════════════════════════
# CẤU HÌNH TRANG VÀ CSS (Giữ nguyên 100% UI từ file Word của bạn)
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="ShoeSenti AI", page_icon="👟", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
html, body, [class*="stApp"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; background: #faf8ff !important; }
[data-testid="stAppViewContainer"], section.main > div { background: #faf8ff !important; }
[data-testid="stVerticalBlock"] { background: transparent !important; }
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span { color: #3b0764 !important; }
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #1e1b4b !important; }

/* ── Gradient Header ── */
.header-gradient {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 25%, #a78bfa 50%, #c084fc 75%, #e879f9 100%);
    padding: 2.2rem 2rem 1.8rem 2rem; border-radius: 0 0 2.5rem 2.5rem; margin: -2rem -2rem 2rem -2rem;
    color: white; text-align: center; position: relative; overflow: hidden; box-shadow: 0 10px 50px rgba(139, 92, 246, 0.25);
}
.header-gradient h1 { color: white !important; font-size: 2.6rem !important; font-weight: 900 !important; margin-bottom: 0.3rem !important; text-shadow: 0 2px 16px rgba(0,0,0,0.12); }
.header-gradient p { color: rgba(255,255,255,0.90) !important; font-size: 1rem !important; margin-top: 0 !important; font-weight: 500; }

/* ── Card chung ── */
.card {
    background: rgba(255, 255, 255, 0.85); backdrop-filter: blur(20px); border-radius: 1.2rem; padding: 1.5rem;
    box-shadow: 0 2px 20px rgba(139, 92, 246, 0.05), 0 1px 3px rgba(0,0,0,0.03); border: 1px solid rgba(255, 255, 255, 0.8); margin-bottom: 1rem;
}
.card h3 { color: #4c1d95; font-weight: 700; font-size: 1.05rem; margin-top: 0 !important; }

/* ── Card kết quả chính ── */
.result-card {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 35%, #a78bfa 65%, #c084fc 100%);
    border-radius: 1.5rem; padding: 2.5rem 2rem; color: white; text-align: center; box-shadow: 0 12px 48px rgba(139, 92, 246, 0.30); margin-bottom: 1rem; position: relative; overflow: hidden;
}
.result-card h2 { color: white !important; font-size: 2.2rem !important; font-weight: 900 !important; margin-bottom: 0.2rem !important; text-shadow: 0 2px 8px rgba(0,0,0,0.12); }
.result-card .emoji-big { font-size: 4rem; margin-bottom: 0.4rem; filter: drop-shadow(0 4px 8px rgba(0,0,0,0.15)); }

/* ── Stat card ── */
.stat-card { background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%); border-radius: 1rem; padding: 1.2rem 1rem; text-align: center; border: 1px solid rgba(139, 92, 246, 0.10); }
.stat-card .stat-value { font-size: 2rem; font-weight: 900; color: #7c3aed; line-height: 1.2; }
.stat-card .stat-label { font-size: 0.8rem; color: #6d28d9; margin-top: 0.3rem; font-weight: 600; }

/* ── Sidebar & Nút bấm ── */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #faf5ff 0%, #f3e8ff 50%, #ede9fe 100%); border-right: 1px solid rgba(139, 92, 246, 0.06); }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a78bfa 100%) !important; color: white !important; border: none !important; border-radius: 0.8rem !important; font-weight: 700 !important; width: 100%; box-shadow: 0 4px 20px rgba(139, 92, 246, 0.30); }
.stDownloadButton > button { border-radius: 0.8rem !important; font-weight: 600 !important; border: 1px solid #ddd6fe !important; }

/* ── Progress bar ── */
.prog-bar-track { background: #ede9fe; border-radius: 1rem; overflow: hidden; height: 0.6rem; margin: 0.3rem 0 0.15rem 0; }
.prog-bar-fill { height: 100%; border-radius: 1rem; transition: width 0.6s ease; }

/* ── History item ── */
.history-item { background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%); border-radius: 0.8rem; padding: 0.75rem 1rem; margin-bottom: 0.5rem; border-left: 4px solid #8b5cf6; font-size: 0.85rem; }

/* ── Các elements khác ── */
.dataframe { border-radius: 0.8rem !important; border: 1px solid #ddd6fe !important; }
.dataframe th { background: #f5f3ff !important; color: #4c1d95 !important; font-weight: 700 !important; }
.stTextArea textarea { border-radius: 0.8rem !important; border: 2px solid #ddd6fe !important; background: #faf5ff !important; color: #1e1b4b !important; }
.stFileUploader { border: 2px dashed #c4b5fd !important; border-radius: 1rem !important; }
.section-divider { height: 1px; background: linear-gradient(90deg, transparent, #c4b5fd, transparent); margin: 1.5rem 0; border: none; }
.empty-state { text-align: center; padding: 2rem; color: #a78bfa; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# CẤU HÌNH MÔ HÌNH VÀ TỪ ĐIỂN
# ═══════════════════════════════════════════════════════════════════════
MODEL_INFO = {
    "name": "ViSoBERT",
    "description": "Mô hình ngôn ngữ tiếng Việt dựa trên BERT",
    "details": "Chuyên biệt cho phân tích cảm xúc đánh giá giày dép đa khía cạnh.",
}

# Tương thích 4 nhãn của ViSoBERT với UI
SENTIMENT_MAP = {0: "None", 1: "Negative", 2: "Positive", 3: "Neutral"}
SENTIMENT_VN = {0: "Không đề cập", 1: "Tiêu cực", 2: "Tích cực", 3: "Trung tính"}
SENTIMENT_EMOJI = {0: "➖", 1: "😡", 2: "😍", 3: "😐"}
SENTIMENT_COLOR = {0: "#9ca3af", 1: "#ef4444", 2: "#10b981", 3: "#8b5cf6"}

ASPECT_TRANSLATION = {
    'Price': 'Giá cả', 'Shipping': 'Vận chuyển', 'Outlook': 'Ngoại hình', 
    'Quality': 'Chất lượng', 'Size': 'Kích cỡ', 'Shop_Service': 'Dịch vụ Shop', 
    'General': 'Tổng quan', 'Others': 'Khác'
}

# ═══════════════════════════════════════════════════════════════════════
# LOAD MODEL VÀ PREDICT BẰNG PYTORCH
# ═══════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = ViSoBERTMultiAspect(MODEL_NAME, NUM_CLASSES)
    
    model_path = "best_visobert_model.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return tokenizer, model, device
    return None, None, None

tokenizer, model, device = load_model()

def predict_visobert(text):
    """Dự đoán bằng ViSoBERT, trả về khía cạnh General cho UI chính, và 8 khía cạnh cho bảng chi tiết"""
    inputs = tokenizer(
        text=text, add_special_tokens=True, max_length=MAX_LEN,
        padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    
    all_aspects_results = {}
    general_result = None
    
    for i, aspect in enumerate(ASPECTS):
        probs = torch.softmax(outputs[i], dim=1).cpu().numpy()[0]
        pred_label = np.argmax(probs)
        confidence = probs[pred_label]
        
        all_aspects_results[aspect] = {
            'label': pred_label,
            'confidence': float(confidence),
            'all_probs': probs.tolist()
        }
        if aspect == "General":
            general_result = all_aspects_results[aspect]
            
    # Nếu câu không có khía cạnh General, lấy khía cạnh có confidence cao nhất làm đại diện
    if general_result['label'] == 0: 
        valid_aspects = [v for k, v in all_aspects_results.items() if v['label'] != 0]
        if valid_aspects:
            general_result = max(valid_aspects, key=lambda x: x['confidence'])

    return general_result, all_aspects_results

def render_progress_bar(label, emoji, pct, color, vn_label):
    return (
        f'<div style="margin-bottom:0.8rem">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.25rem">'
        f'<span style="font-weight:600;font-size:0.95rem">{emoji} {label}</span>'
        f'<span style="font-size:0.85rem;color:#6d28d9">{vn_label} · {pct:.1f}%</span>'
        f'</div>'
        f'<div class="prog-bar-track">'
        f'<div class="prog-bar-fill" style="width:{pct:.1f}%;background:{color}"></div>'
        f'</div></div>'
    )

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Kết quả")
    return output.getvalue()

def to_csv(df):
    return df.to_csv(index=False, encoding="utf-8-sig")

if "history" not in st.session_state:
    st.session_state.history = []

# ═══════════════════════════════════════════════════════════════════════
# GIAO DIỆN CHÍNH
# ═══════════════════════════════════════════════════════════════════════
st.markdown('<div class="header-gradient"><h1>👟 ShoeSenti AI</h1><p>Phân tích cảm xúc đánh giá sản phẩm giày dép bằng ViSoBERT</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div style="text-align:center;margin-bottom:1rem"><div style="font-size:3rem">🧠</div><div style="font-size:1.1rem;font-weight:700;color:#1e1b4b">ShoeSenti AI</div><div style="font-size:0.82rem;color:#7c3aed;margin-top:0.2rem">Sentiment Analysis for Shoes</div></div>', unsafe_allow_html=True)
    with st.expander("🤖 Về mô hình", expanded=True):
        st.markdown(f"**ShoeSenti AI** sử dụng:\n- **{MODEL_INFO['name']}**\n- {MODEL_INFO['details']}")
    with st.expander("🏷️ Nhãn cảm xúc"):
        for label_id in [1, 2, 3]: # Bỏ qua nhãn 0 (Không đề cập) trên UI
            st.markdown(f'<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.6rem"><span style="background:{SENTIMENT_COLOR[label_id]};color:white;padding:0.35rem 1rem;border-radius:1.5rem;font-weight:700;font-size:0.85rem;">{SENTIMENT_EMOJI[label_id]} {SENTIMENT_MAP[label_id]}</span><span style="color:#6d28d9;font-size:0.85rem;font-weight:600">{SENTIMENT_VN[label_id]}</span></div>', unsafe_allow_html=True)
    with st.expander("📂 File hỗ trợ"):
        st.markdown("📄 **CSV** (.csv)  \n📊 **Excel** (.xlsx)  \n📝 Cột văn bản: **Review**")
    st.markdown('<div class="section-divider"></div><div style="text-align:center;font-size:0.78rem;color:#7c3aed;padding:0.5rem 0">Bài tập cuối kỳ Trí tuệ nhân tạo<br>Nhóm 14</div>', unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Không tìm thấy file `best_visobert_model.pt`. Vui lòng kiểm tra lại!")
    st.stop()

tab_single, tab_batch = st.tabs(["📝 Phân tích 1 đánh giá", "📁 Phân tích hàng loạt"])

# ---------------------------------------------------------
# TAB 1: PHÂN TÍCH 1 ĐÁNH GIÁ (Theo thiết kế trong Word)
# ---------------------------------------------------------
with tab_single:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ✍️ Nhập đánh giá giày dép")
    user_input = st.text_area("Nhập đánh giá", placeholder="Ví dụ: Giày đẹp lắm, đi êm", height=140, label_visibility="collapsed")
    analyze_btn = st.button("🔍 Phân tích cảm xúc", key="single", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

    total = len(st.session_state.history)
    if total > 0:
        labels = [h["label"] for h in st.session_state.history if h["label"] != 0]
        most_common = max(set(labels), key=labels.count) if labels else 3
        avg_conf = np.mean([h["confidence"] for h in st.session_state.history])
        s1, s2, s3 = st.columns(3)
        st.markdown(f'<div class="stat-card"><div class="stat-value">{total}</div><div class="stat-label">Tổng phân tích</div></div>', unsafe_allow_html=True)
        with s1: st.markdown(f'<div class="stat-card"><div class="stat-value">{total}</div><div class="stat-label">Tổng phân tích</div></div>', unsafe_allow_html=True)
        with s2: st.markdown(f'<div class="stat-card"><div class="stat-value">{SENTIMENT_EMOJI[most_common]}</div><div class="stat-label">Phổ biến nhất</div></div>', unsafe_allow_html=True)
        with s3: st.markdown(f'<div class="stat-card"><div class="stat-value" style="font-size:1.3rem;color:#7c3aed">{avg_conf:.0f}%</div><div class="stat-label">Độ tin cậy TB</div></div>', unsafe_allow_html=True)

    if analyze_btn and user_input.strip():
        gen_res, full_res = predict_visobert(user_input)
        label_idx = gen_res['label']
        confidence = gen_res['confidence'] * 100
        
        st.session_state.history.insert(0, {"text": user_input[:80], "label": label_idx, "confidence": confidence})

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        res_col1, res_col2 = st.columns([2, 3])

        with res_col1:
            st.markdown(f"""
            <div class="result-card" style="background: linear-gradient(135deg, {SENTIMENT_COLOR[label_idx]} 0%, #8b5cf6 100%);">
                <div class="emoji-big">{SENTIMENT_EMOJI[label_idx]}</div>
                <h2>{SENTIMENT_MAP[label_idx]}</h2>
                <p style="font-size:1.05rem;opacity:0.88;position:relative">{SENTIMENT_VN[label_idx]}</p>
                <div class="confidence">Độ tin cậy: <b>{confidence:.1f}%</b></div>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📈 Xác suất từng lớp")
            # Hiển thị thanh bar cho 3 lớp chính (bỏ qua lớp 0)
            for i in [1, 2, 3]:
                pct = gen_res['all_probs'][i] * 100
                st.markdown(render_progress_bar(SENTIMENT_MAP[i], SENTIMENT_EMOJI[i], pct, SENTIMENT_COLOR[i], SENTIMENT_VN[i]), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        hc1, hc2 = st.columns([5, 1])
        with hc1: st.markdown("### 🕐 Lịch sử phân tích gần đây")
        with hc2:
            if st.button("🗑️ Xóa", key="clear_history"):
                st.session_state.history = []
                st.rerun()
        for i, item in enumerate(st.session_state.history[:5]):
            st.markdown(f'<div class="history-item" style="border-left-color:{SENTIMENT_COLOR[item["label"]]}">{SENTIMENT_EMOJI[item["label"]]} <b>{SENTIMENT_MAP[item["label"]]}</b> <span style="color:#7c3aed;font-size:0.8rem">({item["confidence"]:.1f}%)</span> — <span style="color:#4c1d95">{item["text"]}...</span></div>', unsafe_allow_html=True) 

# ---------------------------------------------------------
# TAB 2: PHÂN TÍCH HÀNG LOẠT (Tích hợp biểu đồ từ Word)
# ---------------------------------------------------------
with tab_batch:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📁 Tải lên file dữ liệu")
    uploaded_file = st.file_uploader("Chọn file CSV hoặc Excel", type=["csv", "xlsx"], label_visibility="collapsed") 

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8-sig") if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
            st.success(f"✅ Đọc thành công: **{len(df):,} dòng**")

            col_sel, col_btn = st.columns([3, 1])
            with col_sel:
                text_col = st.selectbox("📝 Chọn cột văn bản", options=df.columns.tolist(), index=df.columns.tolist().index("Review") if "Review" in df.columns else 0)
            with col_btn:
                st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                analyze_batch_btn = st.button("🔍 Phân tích hàng loạt", key="batch", type="primary")

            if analyze_batch_btn:
                with st.spinner("🔄 Hệ thống ViSoBERT đang bóc tách ngôn ngữ..."):
                    general_preds, general_probs, all_results_list = [], [], []
                    
                    for text in df[text_col].astype(str):
                        gen_res, full_res = predict_visobert(text) 
                        general_preds.append(gen_res['label'])
                        general_probs.append(gen_res['confidence'])
                        
                        # Định dạng dòng kết quả cho bảng chi tiết đa khía cạnh
                        row_res = {}
                        for asp in ASPECTS:
                            row_res[f"{ASPECT_TRANSLATION[asp]}"] = SENTIMENT_VN[full_res[asp]['label']]
                        row_res["Sentiment_Label"] = SENTIMENT_MAP[gen_res['label']]
                        row_res["Confidence"] = f"{gen_res['confidence']*100:.1f}%"
                        all_results_list.append(row_res)

                    df_result = pd.concat([df, pd.DataFrame(all_results_list)], axis=1)
                    
                    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                    st.markdown("## 📊 Kết quả phân tích hàng loạt")

                    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                    dist = pd.Series(general_preds).value_counts()
                    with stat_col1: st.markdown(f'<div class="stat-card"><div class="stat-value">{len(general_preds):,}</div><div class="stat-label">Tổng đánh giá</div></div>', unsafe_allow_html=True) 
                    with stat_col2: st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:#10b981">{dist.get(2, 0):,}</div><div class="stat-label">Tích cực</div></div>', unsafe_allow_html=True) 
                    with stat_col3: st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:#ef4444">{dist.get(1, 0):,}</div><div class="stat-label">Tiêu cực</div></div>', unsafe_allow_html=True) 
                    with stat_col4: st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:#8b5cf6">{np.mean(general_probs)*100:.1f}%</div><div class="stat-label">Độ tin cậy TB</div></div>', unsafe_allow_html=True) 

                    chart_col1, chart_col2 = st.columns(2)
                    # Biểu đồ Cột
                    with chart_col1:
                        st.markdown('<div class="card">### 📊 Phân bố cảm xúc</div>', unsafe_allow_html=True) 
                        fig, ax = plt.subplots(figsize=(5, 3.5))
                        valid_keys = [k for k in sorted(dist.index) if k != 0]
                        bars = ax.bar([SENTIMENT_VN[k] for k in valid_keys], [dist.get(k, 0) for k in valid_keys], color=[SENTIMENT_COLOR[k] for k in valid_keys])
                        for bar in bars: ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(int(bar.get_height())), ha='center', va='bottom')
                        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); fig.patch.set_alpha(0); ax.set_facecolor('none')
                        st.pyplot(fig) 

                    # Biểu đồ Tròn
                    with chart_col2:
                        st.markdown('<div class="card">### 🥧 Tỷ lệ cảm xúc</div>', unsafe_allow_html=True) 
                        fig, ax = plt.subplots(figsize=(4.5, 4.5))
                        ax.pie([dist.get(k, 0) for k in valid_keys], colors=[SENTIMENT_COLOR[k] for k in valid_keys], autopct='%1.1f%%', startangle=140, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
                        ax.legend([f"{SENTIMENT_EMOJI[k]} {SENTIMENT_VN[k]}" for k in valid_keys], loc='lower center', ncol=2, frameon=False)
                        fig.patch.set_alpha(0)
                        st.pyplot(fig) 

                    st.markdown('<div class="card">### 📋 Bảng kết quả chi tiết</div>', unsafe_allow_html=True)
                    st.dataframe(df_result[[text_col, "Sentiment_Label", "Confidence"] + [ASPECT_TRANSLATION[a] for a in ASPECTS]].head(50), use_container_width=True, height=400) 

                    dl_col1, dl_col2 = st.columns(2)
                    with dl_col1: st.download_button("📥 Tải xuống CSV", to_csv(df_result), "visobert_results.csv", "text/csv", use_container_width=True) 
                    with dl_col2: st.download_button("📥 Tải xuống Excel", to_excel(df_result), "visobert_results.xlsx", use_container_width=True) 

        except Exception as e:
            st.error(f"❌ Lỗi: {e}")
    st.markdown('</div>', unsafe_allow_html=True)