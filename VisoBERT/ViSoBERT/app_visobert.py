# -*- coding: utf-8 -*-
"""
ShoeSenti AI — Ứng dụng ViSoBERT phân tích cảm xúc đánh giá giày dép Đa khía cạnh
(Phiên bản UI Hiện đại - Card Design)
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
from io import BytesIO
from transformers import AutoTokenizer

# Bắt buộc: Import class mô hình từ file code của bạn
from Model import ViSoBERTMultiAspect, ASPECTS, MODEL_NAME, MAX_LEN, NUM_CLASSES

# =======================================================================
# CẤU HÌNH TRANG
# =======================================================================
st.set_page_config(
    page_title="ShoeSenti AI",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Chỉnh sửa font chữ tổng thể cho ứng dụng mượt mà hơn
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
html, body, [class*="stApp"] {
    font-family: 'Inter', sans-serif;
}
/* Ẩn bớt khoảng trắng thừa ở trên cùng */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

# =======================================================================
# KHỞI TẠO MÔ HÌNH VÀ TOKENIZER
# =======================================================================
@st.cache_resource
def load_visobert_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = ViSoBERTMultiAspect(MODEL_NAME, NUM_CLASSES)
    
    model_path = "best_visobert_model.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        st.error(f"⚠️ Không tìm thấy file {model_path}. Vui lòng kiểm tra lại!")
        
    model.to(device)
    model.eval()
    return tokenizer, model, device

with st.spinner("⏳ Đang tải 'bộ não' ViSoBERT (1.5GB)... Vui lòng đợi trong giây lát..."):
    tokenizer, model, device = load_visobert_model()

# =======================================================================
# TỪ ĐIỂN VÀ HÀM DỰ ĐOÁN
# =======================================================================
LABEL_MAP = {
    0: {"text": "Không đề cập", "color": "gray", "emoji": "➖"},
    1: {"text": "Tiêu cực", "color": "red", "emoji": "😡"},
    2: {"text": "Tích cực", "color": "green", "emoji": "😍"},
    3: {"text": "Trung tính", "color": "orange", "emoji": "😐"}
}

ASPECT_TRANSLATION = {
    'Price': 'Giá cả', 'Shipping': 'Giao hàng', 'Outlook': 'Ngoại hình', 
    'Quality': 'Chất lượng', 'Size': 'Kích cỡ', 'Shop_Service': 'Dịch vụ Shop', 
    'General': 'Tổng quan', 'Others': 'Khác'
}

def predict_single_review(text):
    inputs = tokenizer(
        text=text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    ).to(device)

    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    
    results = {}
    for i, aspect in enumerate(ASPECTS):
        probs = torch.softmax(outputs[i], dim=1).cpu().numpy()[0]
        pred_label = np.argmax(probs)
        confidence = probs[pred_label]
        
        results[aspect] = {
            'label': pred_label,
            'confidence': float(confidence),
            'all_probs': probs.tolist()
        }
    return results

def to_csv(df):
    return df.to_csv(index=False).encode('utf-8-sig')

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Results')
    return output.getvalue()

# =======================================================================
# GIAO DIỆN CHÍNH (UI REDESIGN)
# =======================================================================

# Header chính
st.markdown("<h1>👟 ShoeSenti AI — Phân Tích Đa Khía Cạnh</h1>", unsafe_allow_html=True)
st.markdown("Hệ thống Trí tuệ Nhân tạo bóc tách chi tiết cảm xúc khách hàng theo 8 tiêu chí.")
st.write("") # Tạo khoảng trống nhỏ

tab1, tab2 = st.tabs(["✍️ Phân tích từng câu", "📁 Phân tích hàng loạt (CSV/Excel)"])

# ---------------------------------------------------------
# TAB 1: PHÂN TÍCH TỪNG CÂU
# ---------------------------------------------------------
with tab1:
    st.write("")
    
    # BOX 1: Vùng nhập dữ liệu (Có viền bo góc)
    with st.container(border=True):
        user_input = st.text_area(
            "Nhập bình luận đánh giá của khách hàng:", 
            height=120,
            placeholder="Ví dụ: Giày đẹp xuất sắc nhưng giao hàng hơi chậm và shop đóng gói sơ sài..."
        )
        
        # Nút bấm kéo dài toàn width của box
        submit_btn = st.button("🚀 Bắt đầu Phân tích", type="primary", use_container_width=True)
    
    # Khi bấm nút phân tích
    if submit_btn:
        if user_input.strip() == "":
            st.warning("Vui lòng nhập nội dung để phân tích!")
        else:
            with st.spinner("Đang bóc tách ngôn ngữ..."):
                preds = predict_single_review(user_input)
            
            st.markdown("### 📊 Kết quả bóc tách:")
            
            # Chia làm 4 cột
            cols = st.columns(4)
            
            for i, aspect in enumerate(ASPECTS):
                col_idx = i % 4
                data = preds[aspect]
                pred_label = data['label']
                conf = data['confidence']
                all_probs = data['all_probs']
                info = LABEL_MAP[pred_label]
                
                with cols[col_idx]:
                    # BOX 2: Viền bo góc cho từng kết quả khía cạnh
                    with st.container(border=True):
                        st.markdown(f"<h4 style='text-align: center; margin-bottom: 0;'>{ASPECT_TRANSLATION[aspect]}</h4>", unsafe_allow_html=True)
                        
                        # In Nhãn và Độ tin cậy ở giữa
                        result_text = f"**{info['emoji']} {info['text']} ({conf*100:.1f}%)**"
                        
                        st.markdown("<div style='text-align: center; padding: 10px 0;'>", unsafe_allow_html=True)
                        if pred_label == 0:
                            st.caption(result_text)
                        elif pred_label == 1:
                            st.error(result_text)
                        elif pred_label == 2:
                            st.success(result_text)
                        else:
                            st.warning(result_text)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Expander xem chi tiết xác suất
                        with st.expander("Phân bổ xác suất"):
                            for label_idx, prob in enumerate(all_probs):
                                label_name = LABEL_MAP[label_idx]['text']
                                safe_prob = max(0.0, min(1.0, float(prob))) 
                                
                                st.caption(f"{label_name}: **{safe_prob*100:.1f}%**")
                                st.progress(safe_prob)

# ---------------------------------------------------------
# TAB 2: PHÂN TÍCH HÀNG LOẠT
# ---------------------------------------------------------
with tab2:
    st.write("")
    with st.container(border=True):
        uploaded_file = st.file_uploader("Tải lên file dữ liệu (.csv, .xlsx)", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                    
                st.success(f"Đã tải thành công {len(df)} dòng dữ liệu!")
                
                text_col = st.selectbox("Chọn cột chứa nội dung Review cần phân tích:", df.columns)
                
                if st.button("⚡ Chạy Mô hình cho toàn bộ File", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    all_results = []
                    total_rows = len(df)
                    
                    for idx, row in df.iterrows():
                        text = str(row[text_col])
                        row_result = {}
                        
                        if pd.isna(text) or text.strip() == "":
                            for asp in ASPECTS:
                                aspect_vi = ASPECT_TRANSLATION[asp]
                                row_result[f"{aspect_vi}_Nhãn"] = "Không đề cập"
                                row_result[f"{aspect_vi}_Độ tin cậy"] = "100.0%"
                        else:
                            preds = predict_single_review(text)
                            for asp, data in preds.items():
                                aspect_vi = ASPECT_TRANSLATION[asp]
                                row_result[f"{aspect_vi}_Nhãn"] = LABEL_MAP[data['label']]['text']
                                row_result[f"{aspect_vi}_Độ tin cậy"] = f"{data['confidence']*100:.2f}%"
                                
                        all_results.append(row_result)
                        
                        if idx % 5 == 0 or idx == total_rows - 1:
                            progress = min((idx + 1) / total_rows, 1.0)
                            progress_bar.progress(progress)
                            status_text.text(f"Đang xử lý: {idx + 1}/{total_rows} dòng...")
                    
                    df_preds = pd.DataFrame(all_results)
                    df_final = pd.concat([df, df_preds], axis=1)
                    
                    status_text.success("✅ Hoàn tất phân tích!")
                    
                    st.markdown("### 📋 Bảng kết quả chi tiết")
                    st.dataframe(df_final.head(50), width=None, height=400)
                    
                    st.markdown("### 📥 Lưu kết quả")
                    dl_col1, dl_col2 = st.columns(2)
                    with dl_col1:
                        st.download_button("📥 Tải xuống CSV", to_csv(df_final), file_name="visobert_results_confidence.csv", mime="text/csv", use_container_width=True)
                    with dl_col2:
                        st.download_button("📥 Tải xuống Excel", to_excel(df_final), file_name="visobert_results_confidence.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
                        
            except Exception as e:
                st.error(f"Có lỗi xảy ra khi đọc file: {e}")