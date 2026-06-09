import streamlit as st
import torch
import torch.nn as nn
import json
from transformers import AutoTokenizer

# ==========================================
# 1. ĐỊNH NGHĨA LẠI KIẾN TRÚC MÔ HÌNH BI-LSTM
# (Bắt buộc phải trùng khớp cấu trúc lúc huấn luyện)
# ==========================================
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        attn_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1) 
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        return context, attn_weights

class RobustSentimentBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=128, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.attention = AttentionLayer(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        embedded = self.embedding(x)           
        lstm_out, _ = self.lstm(embedded) 
        context, attn_weights = self.attention(lstm_out) 
        out = self.classifier(context)                     
        return out

# ==========================================
# 2. TỐI ƯU HÓA: TẢI VÀ CACHE MÔ HÌNH
# ==========================================
@st.cache_resource
def load_model_components():
    # Đọc cấu hình kích thước từ vựng đã lưu
    with open('model_config.json', 'r') as f:
        config = json.load(f)
    vocab_size = config['VOCAB_SIZE']
    
    # Khởi tạo khung mạng và nạp file trọng số .pth
    model = RobustSentimentBiLSTM(vocab_size=vocab_size, embed_dim=256, hidden_dim=128, num_classes=3)
    model.load_state_dict(torch.load('bilstm_model.pth', map_location=torch.device('cpu')))
    model.eval() # Khóa Dropout để phục vụ Inference
    
    # Tải bộ Tokenizer của ViSoBERT để xử lý tiếng Việt ngữ cảnh MXH
    tokenizer = AutoTokenizer.from_pretrained("UET-NLP/VisoBERT")
    
    return model, tokenizer

try:
    model, tokenizer = load_model_components()
except FileNotFoundError:
    st.error("❌ Không tìm thấy file mô hình 'bilstm_model.pth' hoặc 'model_config.json'! Vui lòng chạy file 'train_bilstm.py' trước để sinh file trọng số.")
    st.stop()

# ==========================================
# 3. THIẾT KẾ GIAO DIỆN WEB (UI/UX)
# ==========================================
st.set_page_config(page_title="AI Shoe Sentiment Analysis", page_icon="👟", layout="centered")

# Tiêu đề ứng dụng
st.title("👟 AI Phân tích Cảm xúc Review Giày dép")
st.markdown("""
Hệ thống nhận diện thái độ khách hàng thời gian thực sử dụng kiến trúc học sâu mạng **BiLSTM + Attention Layer**, kết hợp kỹ thuật lọc nhiễu dữ liệu **Heuristic Law**.
""")

# Hộp văn bản nhập bình luận
user_input = st.text_area("Nhập bình luận/đánh giá của khách hàng cần kiểm thử:", 
                          placeholder="Ví dụ: Giày phom đẹp, đi êm chân nhưng shop giao hàng siêu trễ...")

# Xử lý sự kiện nhấn nút dự đoán
if st.button("Phân tích cảm xúc ngay", type="primary"):
    if user_input.strip() == "":
        st.warning("Vui lòng điền nội dung đánh giá trước khi phân tích!")
    else:
        with st.spinner('Trí tuệ nhân tạo đang tính toán toán học ngữ cảnh...'):
            # Tokenize văn bản thô theo đúng chuẩn mã hóa của tập train
            inputs = tokenizer(user_input, padding='max_length', truncation=True, max_length=64, return_tensors="pt")
            input_ids = inputs['input_ids']
            
            # Dự đoán xác suất nhãn thông qua mô hình
            with torch.no_grad():
                outputs = model(input_ids)
                probabilities = torch.softmax(outputs, dim=1)[0]
                predicted_class = torch.argmax(probabilities).item()
            
            # Hiển thị trực quan phân phối xác suất lớp
            st.divider()
            st.subheader("📊 Xác suất phân loại chi tiết:")
            col1, col2, col3 = st.columns(3)
            col1.metric("🔴 Tiêu cực (Lớp 0)", f"{probabilities[0]*100:.2f}%")
            col2.metric("⚪ Trung lập (Lớp 1)", f"{probabilities[1]*100:.2f}%")
            col3.metric("🟢 Tích cực (Lớp 2)", f"{probabilities[2]*100:.2f}%")
            
            # Kết luận nhãn cảm xúc cuối cùng
            st.subheader("🤖 Kết luận từ Mô hình Deep Learning:")
            if predicted_class == 0:
                st.error("📉 **TIÊU CỰC**: Khách hàng đang bày tỏ sự không hài lòng về sản phẩm hoặc dịch vụ.")
            elif predicted_class == 1:
                st.info("⚖️ **TRUNG LẬP**: Đánh giá mang tính mô tả đặc điểm, không mang sắc thái cảm xúc rõ rệt.")
            else:
                st.success("📈 **TÍCH CỰC**: Khách hàng cực kỳ ưng ý và đánh giá cao sản phẩm!")
                st.balloons() # Hiệu ứng bóng bay ăn mừng cho phản hồi tích cực