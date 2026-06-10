import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel,get_linear_schedule_with_warmup
from torch.optim import AdamW
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# ==========================================
# 1. HYPERPARAMETERS (THAM SỐ HUẤN LUYỆN)
# ==========================================
EPOCHS = 10
LEARNING_RATE = 2e-5
# Bạn có thể thay đổi BATCH_SIZE thành 8, 16, 20, hoặc 32 theo ý muốn
BATCH_SIZE = 16 
MAX_LEN = 128
ALPHA = 1.0
GAMMA = 2.0
MAX_GRAD_NORM = 1.0
PATIENCE = 3 # Tham số cho Early Stopping
MODEL_NAME = "uitnlp/visobert"
MODEL_SAVE_PATH = "best_visobert_model.pt"

# Khía cạnh cần đánh giá (8 cột)
ASPECTS = ['Price', 'Shipping', 'Outlook', 'Quality', 'Size', 'Shop_Service', 'General', 'Others']
NUM_CLASSES = 4 # Các nhãn: -1, 0, 1, 2 -> Ánh xạ thành 0, 1, 2, 3

# ==========================================
# 2. XÂY DỰNG DATASET
# ==========================================
class ShoesDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        review = str(self.data.iloc[index]['Review_Cleaned'])
        
        # Lấy nhãn của 8 khía cạnh và cộng 1 để đưa về [0, 1, 2, 3]
        labels = [int(self.data.iloc[index][aspect]) + 1 for aspect in ASPECTS]

        inputs = self.tokenizer(
            text=review,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

# ==========================================
# 3. HÀM MẤT MÁT: FOCAL LOSS
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight # Thêm trọng số để "cứu" các nhãn yếu
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Tính Cross Entropy Loss có tích hợp trọng số
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Áp dụng công thức Focal Loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

# ==========================================
# 4. KIẾN TRÚC MÔ HÌNH: ViSoBERT MULTI-HEAD
# ==========================================
class ViSoBERTMultiAspect(nn.Module):
    def __init__(self, model_name, num_classes):
        super(ViSoBERTMultiAspect, self).__init__()
        self.visobert = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        # Tạo 8 Head (mỗi head dự đoán 4 class cho 1 khía cạnh)
        self.heads = nn.ModuleList([
            nn.Linear(self.visobert.config.hidden_size, num_classes) for _ in range(len(ASPECTS))
        ])

    def forward(self, input_ids, attention_mask):
        outputs = self.visobert(input_ids=input_ids, attention_mask=attention_mask)
        # Sử dụng output của token [CLS]
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0, :]
            
        output = self.drop(pooled_output)
        
        # Trả về list dự đoán cho 8 khía cạnh
        return [head(output) for head in self.heads]

# ==========================================
# 5. HÀM HUẤN LUYỆN VÀ ĐÁNH GIÁ
# ==========================================
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    model.train()
    total_loss = 0

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device) # shape: [batch_size, 8]

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Tính tổng loss của 8 khía cạnh
        loss = 0
        for i in range(len(ASPECTS)):
            loss += loss_fn(outputs[i], labels[:, i])
            
        total_loss += loss.item()
        loss.backward()

        # Chuẩn hóa Gradient (Gradient Clipping)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
        
        optimizer.step()
        scheduler.step()

    return total_loss / len(data_loader)

def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            loss = 0
            for i in range(len(ASPECTS)):
                loss += loss_fn(outputs[i], labels[:, i])
                _, preds = torch.max(outputs[i], dim=1)
                correct_predictions += torch.sum(preds == labels[:, i])
                total_predictions += labels.size(0)

            total_loss += loss.item()

    accuracy = correct_predictions.double() / total_predictions
    return accuracy.item(), total_loss / len(data_loader)

# ==========================================
# 6. CHƯƠNG TRÌNH CHÍNH (MAIN PROCESS)
# ==========================================
def plot_training_history(train_losses, val_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))

    # Biểu đồ Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', marker='o')
    plt.title('Biểu đồ Loss qua các Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Biểu đồ Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy', marker='o')
    plt.title('Biểu đồ Độ chính xác (Accuracy)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    # Lưu biểu đồ thành file ảnh
    plt.savefig('training_graph.png')
    print("\n-> Đã lưu biểu đồ vào file 'training_graph.png'")
    plt.show() # Hiển thị biểu đồ lên màn hình
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")

    # Đọc dữ liệu
 # Tự động lấy thư mục hiện tại của file code đang chạy
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Ghép nối đường dẫn
    train_path = os.path.join(current_dir, "Shoes_Train_Preprocessed_Recovered.csv")
    val_path = os.path.join(current_dir, "Shoes_Validate_Preprocessed_Recovered.csv")

    # Đọc dữ liệu bằng đường dẫn tuyệt đối đã được tự động tạo
    df_train = pd.read_csv(train_path).fillna(" ")
    df_val = pd.read_csv(val_path).fillna(" ")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = ShoesDataset(df_train, tokenizer, MAX_LEN)
    val_dataset = ShoesDataset(df_val, tokenizer, MAX_LEN)

    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ViSoBERTMultiAspect(MODEL_NAME, NUM_CLASSES)
    model = model.to(device)

    # Khởi tạo Optimizer (AdamW)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Số step huấn luyện
    total_steps = len(train_data_loader) * EPOCHS
    
    # Khởi tạo Scheduler (Linear Schedule with Warmup)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps), # 10% warmup
        num_training_steps=total_steps
    )
# Khởi tạo Trọng số Lớp (Class Weights)
    # Thứ tự nhãn trong Pytorch: 0 (None), 1 (Tiêu cực), 2 (Tích cực), 3 (Trung tính)
    # Logic: Nhãn nào xuất hiện ít -> Phạt thật nặng nếu đoán sai (Trọng số cao)
    # Nhãn nào xuất hiện nhiều (như None) -> Trọng số thấp để tránh mô hình bị lười
    
    weights = torch.tensor([
        0.2,  # Nhãn 0 (None): Quá nhiều (~75%), hạ trọng số xuống thấp nhất
        2.5,  # Nhãn 1 (Tiêu cực): Rất ít (~3%), đẩy trọng số lên cao
        0.8,  # Nhãn 2 (Tích cực): Khá nhiều (~17%), giữ ở mức trung bình hơi thấp
        2.5   # Nhãn 3 (Trung tính): Rất ít (~4%) và khó đoán, đẩy trọng số lên CAO NHẤT
    ], dtype=torch.float)
    
    class_weights = weights.to(device)
    # Tăng Gamma từ 2.0 lên 2.5 để ép AI tập trung cực độ vào những câu nó đang đoán sai
    loss_fn = FocalLoss(weight=class_weights, gamma=2.5).to(device)
    # Biến phục vụ Early Stopping
    best_val_accuracy = 0.0
    epochs_no_improve = 0

# === THÊM 3 DÒNG NÀY ĐỂ LƯU TRỮ LỊCH SỬ ===
    history_train_loss = []
    history_val_loss = []
    history_val_acc = []
    print("--- BẮT ĐẦU HUẤN LUYỆN ---")
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 20)

        train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, device, scheduler)
        val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, device)
    
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}')

        # Lưu trữ lịch sử
        history_train_loss.append(train_loss)
        history_val_loss.append(val_loss)
        history_val_acc.append(val_acc)

        # Logic Early Stopping & Save Model
        if val_acc > best_val_accuracy:
            print(f"-> Độ chính xác tăng từ {best_val_accuracy:.4f} lên {val_acc:.4f}. Đang lưu mô hình...")
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"-> Độ chính xác không cải thiện. Counter: {epochs_no_improve}/{PATIENCE}")
            
            if epochs_no_improve >= PATIENCE:
                print(f"!!! Kích hoạt Early Stopping ở epoch {epoch + 1} !!!")
                break
        print()
    # Vẽ biểu đồ sau khi kết thúc huấn luyện
    plot_training_history(history_train_loss, history_val_loss, history_val_acc)
    # 2. Tải lại model tốt nhất vừa lưu để đánh giá chi tiết
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    calculate_detailed_metrics(model, val_data_loader, device)
    print(f"Huấn luyện hoàn tất! Mô hình tốt nhất được lưu tại: {MODEL_SAVE_PATH}")
    
    
def calculate_detailed_metrics(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device) 

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            batch_preds = []
            for i in range(len(ASPECTS)):
                _, preds = torch.max(outputs[i], dim=1)
                batch_preds.append(preds.cpu().numpy())
            
            batch_preds = np.array(batch_preds).T 
            all_preds.extend(batch_preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    # ==========================================
    # CẬP NHẬT: CHỈ ĐÁNH GIÁ TRÊN 3 NHÃN CẢM XÚC (BỎ QUA -1)
    # ==========================================
    # Nhãn trong code: 0 (-1 cũ), 1 (Tiêu cực), 2 (Tích cực), 3 (Trung tính)
    sentiment_labels = [1, 2, 3] 
    sentiment_names = ['Tiêu cực (0)', 'Tích cực (1)', 'Trung tính (2)']
    
    # 1. Bảng báo cáo Text
    print("\n" + "="*60)
    print("BÁO CÁO CHỈ SỐ ĐÁNH GIÁ (ĐÃ LOẠI BỎ NHÃN 'KHÔNG ĐỀ CẬP')")
    print("="*60)
    # Thêm tham số labels=sentiment_labels để ép hệ thống chỉ tính 3 nhãn này
    report = classification_report(all_labels, all_preds, labels=sentiment_labels, target_names=sentiment_names, zero_division=0)
    print(report)
    print("="*60 + "\n")

    # 2. Vẽ Biểu đồ cột (Chỉ cho 3 nhãn)
    report_dict = classification_report(all_labels, all_preds, labels=sentiment_labels, target_names=sentiment_names, zero_division=0, output_dict=True)
    
    precisions = [report_dict[label]['precision'] * 100 for label in sentiment_names]
    recalls = [report_dict[label]['recall'] * 100 for label in sentiment_names]
    f1_scores = [report_dict[label]['f1-score'] * 100 for label in sentiment_names]

    x = np.arange(len(sentiment_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width, precisions, width, label='Precision (%)', color='#4C72B0')
    rects2 = ax.bar(x, recalls, width, label='Recall (%)', color='#DD8452')
    rects3 = ax.bar(x + width, f1_scores, width, label='F1-Score (%)', color='#55A868')

    ax.set_ylabel('Phần trăm (%)', fontsize=12)
    ax.set_title('Sức mạnh Phân loại Cảm xúc Thực tế (Đã loại bỏ nhãn ảo)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sentiment_names, fontsize=12)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax.set_ylim(0, 115)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    ax.bar_label(rects1, padding=3, fmt='%.1f')
    ax.bar_label(rects2, padding=3, fmt='%.1f')
    ax.bar_label(rects3, padding=3, fmt='%.1f')

    fig.tight_layout()
    plt.savefig('real_metrics_bar_chart.png', dpi=300)
    print("-> Đã lưu biểu đồ cột 3 nhãn vào file 'real_metrics_bar_chart.png'")
    plt.close()

    # 3. Vẽ Ma trận nhầm lẫn (Giữ nguyên 4x4 để xem có đoán nhầm cảm xúc thành Không đề cập không)
    target_names_all = ['Không đề cập (-1)', 'Tiêu cực (0)', 'Tích cực (1)', 'Trung tính (2)']
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names_all, yticklabels=target_names_all)
    
    plt.title('Ma trận nhầm lẫn (Toàn cảnh)', fontsize=14, fontweight='bold')
    plt.ylabel('Nhãn Thực tế', fontsize=12)
    plt.xlabel('Nhãn Dự đoán', fontsize=12)
    plt.tight_layout()
    
    plt.savefig('confusion_matrix.png', dpi=300)
    print("-> Đã lưu ảnh Ma trận nhầm lẫn vào file 'confusion_matrix.png'\n")
    plt.close()

    # 4. Tính và hiển thị Độ chính xác tổng thể (Overall Accuracy)
    overall_accuracy = accuracy_score(all_labels, all_preds) * 100
    print("="*60)
    print(f"ĐỘ CHÍNH XÁC TỔNG THỂ (Overall Accuracy): {overall_accuracy:.2f}%")
    print("="*60 + "\n")
if __name__ == "__main__":
    main()