# -*- coding: utf-8 -*-
"""
NAIVE BAYES (BASELINE) — Phân tích cảm xúc đánh giá sản phẩm giày dép
=====================================================================
Mô hình Baseline sử dụng MultinomialNB với TfidfVectorizer / CountVectorizer.

Dữ liệu đầu vào: Các file CSV đã tiền xử lý (Shoes_*_Preprocessed.csv)
Nhãn cảm xúc tổng thể được tổng hợp từ 8 nhãn khía cạnh:
  - -1: Tiêu cực        →  Negative
  -  0: Trung tính       →  Neutral
  -  1: Tích cực         →  Positive
  -  2: Rất tích cực     →  Very Positive

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
from sklearn.naive_bayes import MultinomialNB
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
SENTIMENT_MAP = {0: "Negative", 1: "Neutral", 2: "Positive", 3: "Very Positive"}
SENTIMENT_LABELS = ["Negative", "Neutral", "Positive", "Very Positive"]


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
    
    Quy tắc mới (4 lớp):
      - -1 → 0 (Negative / Tiêu cực)
      -  0 → 1 (Neutral / Trung tính)
      -  1 → 2 (Positive / Tích cực)
      -  2 → 3 (Very Positive / Rất tích cực)
    
    Nhãn tổng thể được tính bằng trung bình tất cả giá trị khía cạnh
    (đã ánh xạ -1→0), sau đó rời rạc hóa:
      - mean < 0.3  → Negative
      - mean < 0.6  → Neutral
      - mean < 1.0  → Positive
      - mean >= 1.0 → Very Positive
    """
    df = df.copy()
    
    # Ánh xạ: -1→0(Negative), 0→1(Neutral), 1→2(Positive), 2→3(Very Positive)
    MAPPING = {-1: 0, 0: 1, 1: 2, 2: 3}
    
    mapped = df[aspect_cols].replace(MAPPING)
    means = mapped.mean(axis=1)
    
    def discretize(val):
        if val < 0.3:
            return 0  # Negative
        elif val < 0.6:
            return 1  # Neutral
        elif val < 1.0:
            return 2  # Positive
        else:
            return 3  # Very Positive
    
    df["Sentiment"] = means.apply(discretize)
    
    return df


# ══════════════════════════════════════════════════════════════════════
# 3. TRÍCH XUẤT ĐẶC TRƯNG & HUẤN LUYỆN
# ══════════════════════════════════════════════════════════════════════

def train_naive_bayes(X_train, y_train, vectorizer_type="tfidf"):
    """
    Huấn luyện mô hình MultinomialNB với Pipeline.
    
    Parameters:
        X_train: Series văn bản đã làm sạch
        y_train: Series nhãn cảm xúc
        vectorizer_type: "tfidf" hoặc "count"
    
    Returns:
        pipeline: Pipeline đã huấn luyện (vectorizer + classifier)
    """
    # token_pattern giữ lại dấu câu quan trọng: , . ! ? ; : - ( )
    # (do tiền xử lý mới đã giữ lại các dấu này)
    # Regex: match từ có thể chứa dấu câu ở cuối, HOẶC dấu câu đứng riêng
    token_pattern = r'(?u)\w+[\w,.!?;:\-]*|[,.!?;:\-()]'

    if vectorizer_type == "tfidf":
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),       # unigram + bigram
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            token_pattern=token_pattern,
        )
        vec_name = "TfidfVectorizer"
    else:
        vectorizer = CountVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            token_pattern=token_pattern,
        )
        vec_name = "CountVectorizer"
    
    pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", MultinomialNB(alpha=1.0)),
    ])
    
    pipeline.fit(X_train, y_train)
    print(f"  ✅ Huấn luyện xong với {vec_name}")
    
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
    labels = [SENTIMENT_MAP.get(k, str(k)) for k in counts.index]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#EF4444", "#9E9E9E", "#4CAF50", "#2196F3"]
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

    emoji_map = {"Negative": "😞", "Neutral": "😐", "Positive": "😊", "Very Positive": "🤩"}
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

    X_train = df_train_full["Review_Cleaned"].astype(str)
    y_train = df_train_full["Sentiment"]
    X_test = df_test["Review_Cleaned"].astype(str)
    y_test = df_test["Sentiment"]

    # Tập Validate dùng để đánh giá overfit
    X_validate = None
    y_validate = None
    if df_validate is not None:
        X_validate = df_validate["Review_Cleaned"].astype(str)
        y_validate = df_validate["Sentiment"]

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
    
    # 3a. TfidfVectorizer + MultinomialNB
    print(f"\n  🔹 Mô hình 1: TfidfVectorizer + MultinomialNB")
    pipeline_tfidf = train_naive_bayes(X_train, y_train, vectorizer_type="tfidf")
    results_tfidf = evaluate_model(pipeline_tfidf, X_test, y_test, model_name="TF-IDF + MultinomialNB")
    print_evaluation(results_tfidf)
    all_results.append(results_tfidf)
    
    # 3b. CountVectorizer + MultinomialNB
    print(f"\n  🔹 Mô hình 2: CountVectorizer + MultinomialNB")
    pipeline_count = train_naive_bayes(X_train, y_train, vectorizer_type="count")
    results_count = evaluate_model(pipeline_count, X_test, y_test, model_name="CountVec + MultinomialNB")
    print_evaluation(results_count)
    all_results.append(results_count)
    
    # ── Đánh giá trên tập Validate để phát hiện overfit ──────────────
    if X_validate is not None and y_validate is not None:
        print(f"\n  ╔{'═' * W}╗")
        print(f"  ║  🔍 ĐÁNH GIÁ OVERFIT TRÊN TẬP VALIDATE{' ' * (W - 44)}║")
        print(f"  ╚{'═' * W}╝")
        
        for pipeline, name in [(pipeline_tfidf, "TF-IDF + MultinomialNB"), (pipeline_count, "CountVec + MultinomialNB")]:
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
        label = SENTIMENT_MAP.get(pred, str(pred))

        # Chọn emoji theo cảm xúc
        emoji_map = {"Negative": "😞", "Neutral": "😐", "Positive": "😊", "Very Positive": "🤩"}
        sent_emoji = emoji_map.get(label, "❓")

        print(f"\n  ┌{'─' * W}┐")
        print(f"  │ 📝 Câu {i}: \"{text}\"")
        print(f"  │")
        print(f"  │ {sent_emoji} Dự đoán: {label}")
        print(f"  │")
        print(f"  │ 📊 Xác suất từng lớp:")
        for j, p in enumerate(prob):
            if j in SENTIMENT_MAP:
                cls_name = SENTIMENT_MAP[j]
                bar_len = int(p * 30)
                bar = "█" * bar_len + "░" * (30 - bar_len)
                print(f"  │    {cls_name:<14} {bar} {p:>6.2%}")
        print(f"  └{'─' * W}┘")

if __name__ == "__main__":
    main()