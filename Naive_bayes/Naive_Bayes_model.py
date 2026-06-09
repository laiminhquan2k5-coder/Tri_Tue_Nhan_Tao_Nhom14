# -*- coding: utf-8 -*-
"""
NAIVE BAYES (BASELINE) — Phân tích cảm xúc đánh giá sản phẩm giày dép
=====================================================================
Mô hình Baseline sử dụng MultinomialNB với TfidfVectorizer / CountVectorizer.

Dữ liệu đầu vào: Các file CSV đã tiền xử lý (Shoes_*_Preprocessed.csv)
Nhãn cảm xúc tổng thể được tổng hợp từ 8 nhãn khía cạnh
(chỉ tính mean trên các khía cạnh có liên quan, bỏ qua giá trị -1):
  - -1: Không liên quan  →  None
  -  0: Tiêu cực          →  Negative
  -  1: Tích cực          →  Positive
  -  2: Trung tính         →  Neutral

Độ đo đánh giá: Accuracy, Precision, Recall, F1-score, Confusion Matrix
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ── Cấu hình đường dẫn ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Nếu script nằm trong Naive_bayes/, DATA_DIR phải quay lên 1 cấp
PARENT_DIR = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == "Naive_bayes" else BASE_DIR
DATA_DIR = os.path.join(PARENT_DIR, "Tri_Tue_Nhan_Tao_Nhom14-main", "Tiền xử lý dữ liệu")
MODEL_DIR = BASE_DIR  # lưu model cùng cấp với script

PREPROCESSED_FILES = {
    "Train":    os.path.join(DATA_DIR, "Shoes_Train_Preprocessed.csv"),
    "Test":     os.path.join(DATA_DIR, "Shoes_Test_Preprocessed.csv"),
    "Validate": os.path.join(DATA_DIR, "Shoes_Validate_Preprocessed.csv"),
}

ASPECT_COLS = ["Price", "Shipping", "Outlook", "Quality", "Size", "Shop_Service", "General", "Others"]

# Nhãn cảm xúc tổng thể (4 lớp)
SENTIMENT_MAP = {-1: "None", 0: "Negative", 1: "Positive", 2: "Neutral"}
SENTIMENT_LABELS = ["None", "Negative", "Positive", "Neutral"]

# Mã hóa nhãn cho sklearn (MultinomialNB yêu cầu nhãn không âm)
ENCODE_MAP = {-1: 0, 0: 1, 1: 2, 2: 3}   # -1→0(None), 0→1(Negative), 1→2(Positive), 2→3(Neutral)
DECODE_MAP = {0: -1, 1: 0, 2: 1, 3: 2}   # Giải mã ngược
# Bản đồ nhãn đã mã hóa (0→None, 1→Negative, 2→Positive, 3→Neutral)
ENCODED_LABEL_MAP = {v: SENTIMENT_MAP[k] for k, v in ENCODE_MAP.items()}


# ══════════════════════════════════════════════════════════════════════
# 1. TẢI DỮ LIỆU
# ══════════════════════════════════════════════════════════════════════

def load_preprocessed_data():
    """Đọc 3 file CSV đã tiền xử lý và trả về dict {name: DataFrame}."""
    data = {}
    for name, path in PREPROCESSED_FILES.items():
        if not os.path.exists(path):
            print(f"  ❌ Không tìm thấy file: {path}")
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        data[name] = df
        print(f"  ✅ Đọc {name}: {len(df):,} dòng, {len(df.columns)} cột")
    return data


# ══════════════════════════════════════════════════════════════════════
# 2. TẠO NHÃN CẢM XÚC TỔNG THỂ
# ══════════════════════════════════════════════════════════════════════

def create_overall_sentiment(df, aspect_cols=ASPECT_COLS):
    """
    Tạo nhãn cảm xúc tổng thể từ 8 nhãn khía cạnh.
    
    Quy tắc (4 lớp):
      - -1 → None (Không liên quan)
      -  0 → Negative (Tiêu cực)
      -  1 → Positive (Tích cực)
      -  2 → Neutral (Trung tính)
    
    Phương pháp: kết hợp quy tắc đa số và mean.
    - Đếm số khía cạnh tích cực (giá trị 1 hoặc 2) và tiêu cực (giá trị 0)
      trong các khía cạnh có liên quan (bỏ qua -1).
    - Nếu không có khía cạnh liên quan → None.
    - Nếu số tích cực > số tiêu cực → Positive.
    - Nếu số tiêu cực > số tích cực → Negative.
    - Nếu bằng nhau, dùng mean để phân biệt:
      + mean >= 1.0 → Positive (nhiều khía cạnh trung tính + tích cực)
      + mean < 0.5  → Negative (nhiều khía cạnh trung tính + tiêu cực)
      + còn lại → Neutral
    """
    df = df.copy()
    
    # Chỉ xét các khía cạnh có liên quan (giá trị != -1)
    relevant_mask = df[aspect_cols] != -1
    relevant_count = relevant_mask.sum(axis=1)
    
    # Đếm số khía cạnh tích cực (giá trị 1 hoặc 2) và tiêu cực (giá trị 0)
    # trong các khía cạnh có liên quan
    positive_count = ((df[aspect_cols] == 1) | (df[aspect_cols] == 2)).sum(axis=1)
    negative_count = (df[aspect_cols] == 0).sum(axis=1)
    
    # Mean chỉ trên các khía cạnh liên quan (bỏ qua -1)
    relevant_sum = df[aspect_cols].where(relevant_mask).sum(axis=1)
    relevant_mean = relevant_sum / relevant_count
    
    def discretize(row):
        pos = row['pos']
        neg = row['neg']
        mean_val = row['mean']
        count = row['count']
        
        if count == 0:
            return -1  # None — không có khía cạnh nào liên quan
        
        # Quy tắc đa số: so sánh số tích cực vs tiêu cực
        if pos > neg:
            return 1  # Positive — nhiều khía cạnh tích cực hơn
        elif neg > pos:
            return 0  # Negative — nhiều khía cạnh tiêu cực hơn
        else:
            # Bằng nhau: dùng mean để phân biệt
            if mean_val >= 1.0:
                return 1  # Positive — mean cao (nhiều giá trị 1 và 2)
            elif mean_val < 0.5:
                return 0  # Negative — mean thấp (nhiều giá trị 0)
            else:
                return 2  # Neutral — pha trộn
    
    df['Sentiment'] = pd.DataFrame({
        'pos': positive_count,
        'neg': negative_count,
        'mean': relevant_mean,
        'count': relevant_count,
    }).apply(discretize, axis=1)
    
    return df


# ══════════════════════════════════════════════════════════════════════
# 3. TRÍCH XUẤT ĐẶC TRƯNG & HUẤN LUYỆN
# ══════════════════════════════════════════════════════════════════════

def train_naive_bayes(X_train, y_train, vectorizer_type="tfidf", classifier_type="multinomial", use_balanced=True):
    """
    Huấn luyện mô hình Naive Bayes với Pipeline.
    
    Parameters:
        X_train: Series văn bản đã làm sạch
        y_train: Series nhãn cảm xúc
        vectorizer_type: "tfidf" hoặc "count"
        classifier_type: "multinomial" hoặc "complement"
        use_balanced: Có dùng sample_weight để cân bằng lớp hay không
    
    Returns:
        pipeline: Pipeline đã huấn luyện (vectorizer + classifier)
    """
    # token_pattern giữ lại dấu câu quan trọng: , . ! ? ; : - ( )
    token_pattern = r'(?u)\w+[\w,.!?;:\-]*|[,.!?;:\-()]'

    if vectorizer_type == "tfidf":
        vectorizer = TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            token_pattern=token_pattern,
        )
        vec_name = "TfidfVectorizer"
    else:
        vectorizer = CountVectorizer(
            max_features=15000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            token_pattern=token_pattern,
        )
        vec_name = "CountVectorizer"
    
    if classifier_type == "complement":
        classifier = ComplementNB(alpha=0.5, norm=True)
        cls_name = "ComplementNB"
    else:
        classifier = MultinomialNB(alpha=0.5)
        cls_name = "MultinomialNB"
    
    pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", classifier),
    ])
    
    # Cân bằng lớp bằng sample_weight
    if use_balanced:
        sample_weight = compute_sample_weight("balanced", y_train)
        pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weight)
    else:
        pipeline.fit(X_train, y_train)
    
    print(f"  ✅ Huấn luyện xong với {vec_name} + {cls_name} (balanced={use_balanced})")
    
    return pipeline


# ══════════════════════════════════════════════════════════════════════
# 4. ĐÁNH GIÁ MÔ HÌNH
# ══════════════════════════════════════════════════════════════════════

def evaluate_model(pipeline, X_test, y_test, model_name="Model"):
    """
    Đánh giá mô hình với 5 độ đo: Accuracy, Precision, Recall, F1, Confusion Matrix.
    Trả về dict chứa tất cả kết quả.
    """
    y_pred = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])
    # Encoded labels: 0=None, 1=Negative, 2=Positive, 3=Neutral
    report = classification_report(
        y_test, y_pred,
        target_names=SENTIMENT_LABELS,
        zero_division=0,
    )
    
    results = {
        "model_name": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_pred": y_pred,
        "y_test": y_test,
    }
    
    return results


def print_evaluation(results):
    """In kết quả đánh giá ra console với định dạng đẹp."""
    name = results['model_name']
    W = 60  # chiều rộng khung

    # ── Header ──
    print(f"\n  ╔{'═' * W}╗")
    print(f"  ║  📊 KẾT QUẢ ĐÁNH GIÁ: {name:<{W - 28}}║")
    print(f"  ╠{'═' * W}╣")

    # ── 4.1.1–4.1.4 Các độ đo tổng hợp ──
    metrics_items = [
        ("4.1.1  Accuracy ", f"{results['accuracy']:.4f}  ({results['accuracy']*100:.2f}%)", "🎯"),
        ("4.1.2  Precision", f"{results['precision']:.4f}", "🎯"),
        ("4.1.3  Recall   ", f"{results['recall']:.4f}", "🎯"),
        ("4.1.4  F1-score ", f"{results['f1_score']:.4f}", "🎯"),
    ]
    for label, value, icon in metrics_items:
        row = f"  {icon} {label}:  {value}"
        print(f"  ║  {row:<{W - 2}}║")
    print(f"  ╠{'═' * W}╣")

    # ── 4.1.5 Classification Report (bảng đẹp) ──
    print(f"  ║  📋 4.1.5 Classification Report:{' ' * (W - 33)}║")
    print(f"  ╠{'═' * W}╣")

    report = classification_report(
        results["y_test"], results["y_pred"],
        target_names=SENTIMENT_LABELS,
        output_dict=True, zero_division=0,
    )

    # Header row
    hdr = f"  ║  {'Lớp':<16}│{'Precision':>10}│{'Recall':>10}│{'F1-score':>10}│{'Support':>10} ║"
    sep = f"  ║  {'─' * 16}┼{'─' * 10}┼{'─' * 10}┼{'─' * 10}┼{'─' * 10} ║"
    print(hdr)
    print(sep)

    for cls in SENTIMENT_LABELS:
        d = report[cls]
        row = f"  ║  {cls:<16}│{d['precision']:>10.4f}│{d['recall']:>10.4f}│{d['f1-score']:>10.4f}│{int(d['support']):>10} ║"
        print(row)
    print(sep)

    # Weighted avg
    wa = report["weighted avg"]
    row_w = f"  ║  {'Weighted Avg':<16}│{wa['precision']:>10.4f}│{wa['recall']:>10.4f}│{wa['f1-score']:>10.4f}│{int(wa['support']):>10} ║"
    print(row_w)
    # Macro avg
    ma = report["macro avg"]
    row_m = f"  ║  {'Macro Avg':<16}│{ma['precision']:>10.4f}│{ma['recall']:>10.4f}│{ma['f1-score']:>10.4f}│{int(ma['support']):>10} ║"
    print(row_m)
    print(f"  ╠{'═' * W}╣")

    # ── Confusion Matrix (bảng đẹp) ──
    cm = results["confusion_matrix"]
    print(f"  ║  📊 Confusion Matrix:{' ' * (W - 24)}║")
    print(f"  ╠{'═' * W}╣")

    # Header
    cm_hdr = "".join([f"{lbl:>14}" for lbl in SENTIMENT_LABELS])
    print(f"  ║  {'Dự đoán →':<14}{cm_hdr}  ║")
    print(f"  ║  {'─' * 14}{'─' * 14 * len(SENTIMENT_LABELS)}  ║")

    for i, lbl in enumerate(SENTIMENT_LABELS):
        vals = "".join([f"{cm[i][j]:>14d}" for j in range(len(SENTIMENT_LABELS))])
        print(f"  ║  {lbl:<14}{vals}  ║")
    print(f"  ╚{'═' * W}╝")


# ══════════════════════════════════════════════════════════════════════
# 5. VẼ BIỂU ĐỒ
# ══════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(cm, model_name, save_path=None):
    """Vẽ Confusion Matrix dưới dạng heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=SENTIMENT_LABELS,
        yticklabels=SENTIMENT_LABELS,
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
    )
    ax.set_xlabel("Dự đoán (Predicted)", fontsize=12, labelpad=10)
    ax.set_ylabel("Thực tế (Actual)", fontsize=12, labelpad=10)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  📊 Lưu biểu đồ: {save_path}")
    plt.close()


def plot_metrics_comparison(all_results, save_path=None):
    """Vẽ biểu đồ so sánh các độ đo giữa các mô hình."""
    model_names = [r["model_name"] for r in all_results]
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    
    x = np.arange(len(metric_labels))
    width = 0.3
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]
    
    for i, result in enumerate(all_results):
        values = [result[m] for m in metrics]
        bars = ax.bar(
            x + i * width, values, width,
            label=result["model_name"],
            color=colors[i % len(colors)],
            alpha=0.85,
            edgecolor="white",
            linewidth=1,
        )
        # Thêm giá trị lên mỗi cột
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold",
            )
    
    ax.set_xlabel("Độ đo (Metric)", fontsize=12, labelpad=10)
    ax.set_ylabel("Giá trị (Score)", fontsize=12, labelpad=10)
    ax.set_title("So sánh các độ đo đánh giá mô hình Naive Bayes", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x + width * (len(all_results) - 1) / 2)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  📊 Lưu biểu đồ: {save_path}")
    plt.close()


def plot_sentiment_distribution(y_series, title, save_path=None):
    """Vẽ biểu đồ phân bố nhãn cảm xúc."""
    counts = y_series.value_counts().sort_index()
    labels = [ENCODED_LABEL_MAP.get(k, str(k)) for k in counts.index]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#9E9E9E", "#EF4444", "#4CAF50", "#2196F3"]
    bars = ax.bar(labels, counts.values, color=colors[:len(labels)], edgecolor="white", linewidth=1.5)
    
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(val),
            ha="center", va="bottom",
            fontsize=11, fontweight="bold",
        )
    
    ax.set_xlabel("Cảm xúc (Sentiment)", fontsize=12, labelpad=10)
    ax.set_ylabel("Số lượng (Count)", fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  📊 Lưu biểu đồ: {save_path}")
    plt.close()


def plot_per_class_metrics(results, save_path=None):
    """Vẽ biểu đồ Precision/Recall/F1 theo từng class."""
    report = classification_report(
        results["y_test"], results["y_pred"],
        target_names=SENTIMENT_LABELS,
        output_dict=True,
        zero_division=0,
    )
    
    classes = SENTIMENT_LABELS
    metrics_names = ["precision", "recall", "f1-score"]
    
    data = {}
    for cls in classes:
        data[cls] = [report[cls][m] for m in metrics_names]
    
    x = np.arange(len(classes))
    width = 0.25
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, metric in enumerate(metrics_names):
        values = [data[cls][i] for cls in classes]
        bars = ax.bar(
            x + i * width, values, width,
            label=metric.capitalize(),
            color=colors[i],
            alpha=0.85,
            edgecolor="white",
            linewidth=1,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold",
            )
    
    ax.set_xlabel("Lớp cảm xúc (Sentiment Class)", fontsize=12, labelpad=10)
    ax.set_ylabel("Giá trị (Score)", fontsize=12, labelpad=10)
    ax.set_title(f"Precision / Recall / F1 theo từng lớp — {results['model_name']}", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  📊 Lưu biểu đồ: {save_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# 6. LƯU MÔ HÌNH
# ══════════════════════════════════════════════════════════════════════

def save_model(pipeline, model_name, model_dir=MODEL_DIR):
    """Lưu mô hình Pipeline (vectorizer + classifier) bằng pickle."""
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"{model_name}.pkl")
    
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    
    print(f"  💾 Lưu mô hình: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════
# 7. HÀM MAIN — CHẠY TOÀN BỘ PIPELINE
# ══════════════════════════════════════════════════════════════════════

def main():
    W = 70
    print(f"\n  ╔{'═' * W}╗")
    print(f"  ║  🤖 NAIVE BAYES (BASELINE) — PHÂN TÍCH CẢM XÚC ĐÁNH GIÁ GIÀY DÉP{' ' * (W - 65)}║")
    print(f"  ╚{'═' * W}╝")

    # ── Bước 1: Tải dữ liệu ─────────────────────────────────────────
    print(f"\n  ╔{'═' * W}╗")
    print(f"  ║  📂 BƯỚC 1: TẢI DỮ LIỆU ĐÃ TIỀN XỬ LÝ{' ' * (W - 42)}║")
    print(f"  ╚{'═' * W}╝")
    
    data = load_preprocessed_data()
    
    if not data:
        print("  ❌ Không có dữ liệu. Kết thúc.")
        return
    
    # ── Bước 2: Tạo nhãn cảm xúc tổng thể ────────────────────────────
    print(f"\n  ╔{'═' * W}╗")
    print(f"  ║  🏷️ BƯỚC 2: TẠO NHÃN CẢM XÚC TỔNG THỂ{' ' * (W - 44)}║")
    print(f"  ╚{'═' * W}╝")

    emoji_map = {"None": "🚫", "Negative": "😞", "Positive": "😊", "Neutral": "😐"}
    for name, df in data.items():
        print(f"\n  📂 {name}:")
        data[name] = create_overall_sentiment(df)
        sentiment_counts = data[name]["Sentiment"].value_counts().sort_index()
        total = sentiment_counts.sum()
        for label, count in sentiment_counts.items():
            label_name = SENTIMENT_MAP.get(label, str(label))
            pct = count / total * 100
            bar_len = int(pct / 100 * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            emoji = emoji_map.get(label_name, "")
            print(f"     {emoji} {label_name:<14} {bar} {count:>5,} ({pct:5.1f}%)")
    
    # Tách tập train/test/validate
    df_train = data.get("Train")
    df_test = data.get("Test")
    df_validate = data.get("Validate")
    
    if df_train is None or df_test is None:
        print("  ❌ Thiếu tập Train hoặc Test. Kết thúc.")
        return
    
    # Không gộp Train + Validate — giữ Validate riêng để phát hiện overfit
    df_train_full = df_train

    # ── Oversampling lớp None (quá hiếm: chỉ ~0.1%) ──
    # Nhân bản mẫu None lên ít nhất 5% tổng dữ liệu để mô hình học được
    none_mask = df_train_full["Sentiment"] == -1
    none_count = none_mask.sum()
    total_count = len(df_train_full)
    target_none = max(int(total_count * 0.05), 200)  # ít nhất 5% hoặc 200 mẫu
    if none_count > 0 and none_count < target_none:
        df_none = df_train_full[none_mask]
        df_other = df_train_full[~none_mask]
        df_none_upsampled = resample(
            df_none,
            replace=True,
            n_samples=target_none - none_count,
            random_state=42,
        )
        df_train_full = pd.concat([df_train_full, df_none_upsampled], ignore_index=True)
        print(f"\n  🔄 Oversampling lớp None: {none_count} → {none_count + (target_none - none_count)} mẫu")
        print(f"  📊 Tổng dữ liệu huấn luyện: {total_count} → {len(df_train_full)} mẫu")

    X_train = df_train_full["Review_Cleaned"].astype(str)
    y_train = df_train_full["Sentiment"].map(ENCODE_MAP)
    X_test = df_test["Review_Cleaned"].astype(str)
    y_test = df_test["Sentiment"].map(ENCODE_MAP)

    # Tập Validate dùng để đánh giá overfit
    X_validate = None
    y_validate = None
    if df_validate is not None:
        X_validate = df_validate["Review_Cleaned"].astype(str)
        y_validate = df_validate["Sentiment"].map(ENCODE_MAP)

    print(f"\n  📊 Tập huấn luyện: {len(X_train):,} mẫu")
    print(f"  📊 Tập kiểm thử:   {len(X_test):,} mẫu")
    if X_validate is not None:
        print(f"  📊 Tập validate:   {len(X_validate):,} mẫu")
    
    # ── Vẽ phân bố nhãn ──────────────────────────────────────────────
    plot_sentiment_distribution(
        y_train,
        "Phân bố nhãn cảm xúc — Tập huấn luyện",
        save_path=os.path.join(MODEL_DIR, "sentiment_distribution_train.png"),
    )
    plot_sentiment_distribution(
        y_test,
        "Phân bố nhãn cảm xúc — Tập kiểm thử",
        save_path=os.path.join(MODEL_DIR, "sentiment_distribution_test.png"),
    )
    
    # ── Bước 3: Huấn luyện mô hình ──────────────────────────────────
    print(f"\n  ╔{'═' * W}╗")
    print(f"  ║  🧠 BƯỚC 3: HUẤN LUYỆN MÔ HÌNH NAIVE BAYES{' ' * (W - 48)}║")
    print(f"  ╚{'═' * W}╝")
    
    all_results = []
    
    # 3a. TfidfVectorizer + MultinomialNB (balanced)
    print(f"\n  🔹 Mô hình 1: TfidfVectorizer + MultinomialNB (balanced)")
    pipeline_tfidf = train_naive_bayes(X_train, y_train, vectorizer_type="tfidf", classifier_type="multinomial", use_balanced=True)
    results_tfidf = evaluate_model(pipeline_tfidf, X_test, y_test, model_name="TF-IDF + MultinomialNB")
    print_evaluation(results_tfidf)
    all_results.append(results_tfidf)
    
    # 3b. CountVectorizer + MultinomialNB (balanced)
    print(f"\n  🔹 Mô hình 2: CountVectorizer + MultinomialNB (balanced)")
    pipeline_count = train_naive_bayes(X_train, y_train, vectorizer_type="count", classifier_type="multinomial", use_balanced=True)
    results_count = evaluate_model(pipeline_count, X_test, y_test, model_name="CountVec + MultinomialNB")
    print_evaluation(results_count)
    all_results.append(results_count)
    
    # 3c. TfidfVectorizer + ComplementNB (balanced)
    print(f"\n  🔹 Mô hình 3: TfidfVectorizer + ComplementNB (balanced)")
    pipeline_tfidf_comp = train_naive_bayes(X_train, y_train, vectorizer_type="tfidf", classifier_type="complement", use_balanced=True)
    results_tfidf_comp = evaluate_model(pipeline_tfidf_comp, X_test, y_test, model_name="TF-IDF + ComplementNB")
    print_evaluation(results_tfidf_comp)
    all_results.append(results_tfidf_comp)
    
    # 3d. CountVectorizer + ComplementNB (balanced)
    print(f"\n  🔹 Mô hình 4: CountVectorizer + ComplementNB (balanced)")
    pipeline_count_comp = train_naive_bayes(X_train, y_train, vectorizer_type="count", classifier_type="complement", use_balanced=True)
    results_count_comp = evaluate_model(pipeline_count_comp, X_test, y_test, model_name="CountVec + ComplementNB")
    print_evaluation(results_count_comp)
    all_results.append(results_count_comp)
    
    # ── Đánh giá trên tập Validate để phát hiện overfit ──────────────
    if X_validate is not None and y_validate is not None:
        print(f"\n  ╔{'═' * W}╗")
        print(f"  ║  🔍 ĐÁNH GIÁ OVERFIT TRÊN TẬP VALIDATE{' ' * (W - 44)}║")
        print(f"  ╚{'═' * W}╝")
        
        for pipeline, name in [(pipeline_tfidf, "TF-IDF + MultinomialNB"), (pipeline_count, "CountVec + MultinomialNB"), (pipeline_tfidf_comp, "TF-IDF + ComplementNB"), (pipeline_count_comp, "CountVec + ComplementNB")]:
            val_pred = pipeline.predict(X_validate)
            val_acc = accuracy_score(y_validate, val_pred)
            val_f1 = f1_score(y_validate, val_pred, average="weighted", zero_division=0)
            train_pred = pipeline.predict(X_train)
            train_acc = accuracy_score(y_train, train_pred)
            train_f1 = f1_score(y_train, train_pred, average="weighted", zero_division=0)
            test_f1 = next(r["f1_score"] for r in all_results if r["model_name"] == name)
            gap = train_f1 - test_f1
            
            print(f"\n  📊 {name}:")
            print(f"     Train F1:     {train_f1:.4f}")
            print(f"     Test F1:      {test_f1:.4f}")
            print(f"     Validate F1:  {val_f1:.4f}")
            print(f"     Gap (Train-Test): {gap:.4f}", end="")
            if gap > 0.15:
                print(f"  ⚠️  CÓ DẤU HIỆU OVERFIT (gap > 0.15)")
            elif gap > 0.08:
                print(f"  ⚠️  Overfit nhẹ (gap > 0.08)")
            else:
                print(f"  ✅ Không overfit")
    
    # ── Bước 4: Vẽ biểu đồ đánh giá ──────────────────────────────────
    print(f"\n  ╔{'═' * W}╗")
    print(f"  ║  📈 BƯỚC 4: VẼ BIỂU ĐỒ ĐÁNH GIÁ{' ' * (W - 38)}║")
    print(f"  ╚{'═' * W}╝")
    
    # Confusion Matrix cho từng mô hình
    for r in all_results:
        safe_name = r["model_name"].replace(" ", "_").replace("+", "plus")
        plot_confusion_matrix(
            r["confusion_matrix"],
            r["model_name"],
            save_path=os.path.join(MODEL_DIR, f"confusion_matrix_{safe_name}.png"),
        )
    
    # So sánh các độ đo giữa 2 mô hình
    plot_metrics_comparison(
        all_results,
        save_path=os.path.join(MODEL_DIR, "metrics_comparison.png"),
    )
    
    # Precision/Recall/F1 theo từng class cho mô hình tốt nhất
    best_result = max(all_results, key=lambda r: r["f1_score"])
    print(f"\n  🏆 Mô hình tốt nhất (F1): {best_result['model_name']} (F1 = {best_result['f1_score']:.4f})")
    
    safe_best = best_result["model_name"].replace(" ", "_").replace("+", "plus")
    plot_per_class_metrics(
        best_result,
        save_path=os.path.join(MODEL_DIR, f"per_class_metrics_{safe_best}.png"),
    )
    
    # ── Bước 5: Lưu mô hình ──────────────────────────────────────────
    print(f"\n  ╔{'═' * W}╗")
    print(f"  ║  💾 BƯỚC 5: LƯU MÔ HÌNH (.pkl){' ' * (W - 34)}║")
    print(f"  ╚{'═' * W}╝")
    
    save_model(pipeline_tfidf, "naive_bayes_tfidf")
    save_model(pipeline_count, "naive_bayes_countvec")
    save_model(pipeline_tfidf_comp, "naive_bayes_tfidf_complement")
    save_model(pipeline_count_comp, "naive_bayes_countvec_complement")
    
    # ── Bảng tổng kết ────────────────────────────────────────────────
    W = 70
    print(f"\n  ╔{'═' * W}╗")
    print(f"  ║  📋 BẢNG TỔNG KẾT{' ' * (W - 20)}║")
    print(f"  ╠{'═' * W}╣")

    # Header
    hdr = f"  ║  {'Mô hình':<26}│{'Accuracy':>9}│{'Precision':>9}│{'Recall':>9}│{'F1-score':>9} ║"
    sep = f"  ║  {'─' * 26}┼{'─' * 9}┼{'─' * 9}┼{'─' * 9}┼{'─' * 9} ║"
    print(hdr)
    print(sep)

    for r in all_results:
        name = r["model_name"]
        is_best = (r is best_result)
        marker = " 🏆" if is_best else "  "
        row = f"  ║  {name:<26}│{r['accuracy']:>9.4f}│{r['precision']:>9.4f}│{r['recall']:>9.4f}│{r['f1_score']:>9.4f} {marker}║"
        print(row)
    print(f"  ╚{'═' * W}╝")

    print(f"\n  ╔{'═' * W}╗")
    print(f"  ║  ✅ HOÀN TẤT!{' ' * (W - 15)}║")
    print(f"  ╠{'═' * W}╣")
    print(f"  ║  📁 Mô hình đã lưu tại: {MODEL_DIR:<{W - 29}}║")
    print(f"  ║  📁 Biểu đồ đã lưu tại: {MODEL_DIR:<{W - 29}}║")
    print(f"  ╚{'═' * W}╝")
    
    # ── Demo dự đoán nhanh ──────────────────────────────────────────
    W = 70
    print(f"\n  ╔{'═' * W}╗")
    print(f"  ║  🔮 DEMO: DỰ ĐOÁN CẢM XÚC VỚI MÔ HÌNH TỐT NHẤT{' ' * (W - 53)}║")
    print(f"  ║  🏆 Mô hình: {best_result['model_name']}{' ' * (W - 16 - len(best_result['model_name']))}║")
    print(f"  ╚{'═' * W}╝")

    demo_texts = [
        "giày đẹp đi êm lắm rất hài lòng",
        "chất lượng bình thường giao hàng chậm",
        "sản phẩm tuyệt vời giá rẻ đáng mua",
        "đế giày bị hỏng shop phục vụ kém",
    ]

    best_pipeline = pipeline_tfidf if best_result["model_name"].startswith("TF-IDF") else pipeline_count

    for i, text in enumerate(demo_texts, 1):
        pred = best_pipeline.predict([text])[0]
        prob = best_pipeline.predict_proba([text])[0]
        pred_original = DECODE_MAP.get(pred, pred)
        label = SENTIMENT_MAP.get(pred_original, str(pred_original))

        # Chọn emoji theo cảm xúc
        emoji_map = {"None": "🚫", "Negative": "😞", "Positive": "😊", "Neutral": "😐"}
        sent_emoji = emoji_map.get(label, "❓")

        print(f"\n  ┌{'─' * W}┐")
        print(f"  │ 📝 Câu {i}: \"{text}\"")
        print(f"  │")
        print(f"  │ {sent_emoji} Dự đoán: {label}")
        print(f"  │")
        print(f"  │ 📊 Xác suất từng lớp:")
        for j, p in enumerate(prob):
            j_original = DECODE_MAP.get(j, j)
            if j_original in SENTIMENT_MAP:
                cls_name = SENTIMENT_MAP[j_original]
                bar_len = int(p * 30)
                bar = "█" * bar_len + "░" * (30 - bar_len)
                print(f"  │    {cls_name:<14} {bar} {p:>6.2%}")
        print(f"  └{'─' * W}┘")

if __name__ == "__main__":
    main()