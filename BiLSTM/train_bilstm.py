import pandas as pd
import numpy as np
import ast
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import json
import os

# ==========================================
# 1. TIỀN XỬ LÝ & LỌC NHÃN RÁC (HEURISTIC)
# ==========================================
print("--- ĐANG CHẠY LUẬT HEURISTIC LÀM SẠCH VÀ VẮT KIỆT DỮ LIỆU ---")
files = ['BiLSTM\\Shoes_Train_Preprocessed.csv', 'BiLSTM\\Shoes_Validate_Preprocessed.csv', 'BiLSTM\\Shoes_Test_Preprocessed.csv']
aspects = ['Price', 'Shipping', 'Outlook', 'Quality', 'Size', 'Shop_Service']

def apply_heuristic(row):
    if row['General'] != -1: return row['General']
    vals = [row[asp] for asp in aspects if row[asp] != -1]
    if len(vals) == 0: return -1
    if 0 in vals: return 0
    elif all(v == 2 for v in vals): return 2
    elif all(v in [1, 2] for v in vals): return 1
    return -1

for f in files:
    if os.path.exists(f):
        df = pd.read_csv(f)
        df['General'] = df.apply(apply_heuristic, axis=1)
        df_clean = df[df['General'] != -1].copy()
        df_clean.to_csv(f.replace('.csv', '_Augmented.csv'), index=False)
    else:
        raise FileNotFoundError(f"Không tìm thấy file {f} trong thư mục hiện tại! Vui lòng kiểm tra lại cấu trúc.")
print("-> Đã tạo xong các file dữ liệu làm sạch (*_Augmented.csv)!\n")

# ==========================================
# 2. CẤU HÌNH THÔNG SỐ VÀ DATA LOADER
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 8  # Bạn có thể tăng lên nếu muốn độ chính xác cao hơn nữa
LEARNING_RATE = 5e-4
TARGET_COL = 'General'

class AugmentedShoeDataset(Dataset):
    def __init__(self, file_path, max_len=64):
        df = pd.read_csv(file_path)
        self.labels = df[TARGET_COL].values
        self.input_ids = []
        for ids_str in df['input_ids']:
            ids = ast.literal_eval(ids_str) if isinstance(ids_str, str) else ids_str
            # Padding tự động bằng token ID = 1 (Chuẩn của VisoBERT)
            ids = ids[:max_len] + [1] * max(0, max_len - len(ids))
            self.input_ids.append(ids)
        self.input_ids = torch.tensor(self.input_ids, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.input_ids[idx], self.labels[idx]

train_dataset = AugmentedShoeDataset('BiLSTM\\Shoes_Train_Preprocessed_Augmented.csv')
valid_dataset = AugmentedShoeDataset('BiLSTM\\Shoes_Validate_Preprocessed_Augmented.csv')
test_dataset = AugmentedShoeDataset('BiLSTM\\Shoes_Test_Preprocessed_Augmented.csv')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Tính toán kích thước từ vựng tối đa động
VOCAB_SIZE = max(train_dataset.input_ids.max().item(), valid_dataset.input_ids.max().item(), test_dataset.input_ids.max().item()) + 500

# Lưu Vocab Size lại làm cấu hình cho Streamlit
with open('BiLSTM\\model_config.json', 'w') as f:
    json.dump({'VOCAB_SIZE': VOCAB_SIZE}, f)

# Tính trọng số phạt lớp mất cân bằng
class_counts = np.bincount(train_dataset.labels.numpy())
class_weights = len(train_dataset) / (3 * class_counts)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

# ==========================================
# 3. KIẾN TRÚC MÔ HÌNH BiLSTM + ATTENTION (Giữ nguyên gốc của bạn)
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

model = RobustSentimentBiLSTM(vocab_size=VOCAB_SIZE).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# ==========================================
# 4. HUẤN LUYỆN VÀ TRÍCH XUẤT FILE MÔ HÌNH
# ==========================================
best_val_loss = float('inf')
patience_counter, EARLY_STOPPING_PATIENCE = 0, 3
best_model_wts = copy.deepcopy(model.state_dict())

print("\n--- BẮT ĐẦU TRAINING MÔ HÌNH TRÊN VS CODE ---")
for epoch in range(EPOCHS):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    model.eval()
    total_valid_loss = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            total_valid_loss += criterion(model(inputs), labels).item()
            
    avg_valid_loss = total_valid_loss / len(valid_loader)
    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Valid Loss: {avg_valid_loss:.4f}")
    
    scheduler.step(avg_valid_loss)
    if avg_valid_loss < best_val_loss:
        best_val_loss = avg_valid_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(">>> Kích hoạt Early Stopping để tránh Overfitting!")
            break

# LƯU TRỌNG SỐ MÔ HÌNH LẠI THƯ MỤC GỐC
torch.save(best_model_wts, 'BiLSTM\\bilstm_model.pth')
print("\n🎉 XUẤT FILE MÔ HÌNH THÀNH CÔNG! Đã tạo ra file 'bilstm_model.pth' và 'model_config.json'")