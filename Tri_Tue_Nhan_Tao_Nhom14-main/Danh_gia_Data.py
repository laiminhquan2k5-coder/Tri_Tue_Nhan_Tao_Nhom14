"""
Phép đo độ đồng thuận Am (Agreement Measure) — Siegel & Castellan (1988)

Đo mức độ đồng thuận giữa các nhãn gán trong bộ dữ liệu comment Shopee về giày.
Bộ dữ liệu gồm 8 nhãn: Price, Shipping, Outlook, Quality, Size, Shop_Service, General, Others
Mỗi nhãn có 4 giá trị cảm xúc:
  - -1: Không liên quan (khía cạnh không xuất hiện trong comment)
  -  0: Tiêu cực (nhận xét tiêu cực về khía cạnh đó)
  -  1: Tích cực (nhận xét tích cực về khía cạnh đó)
  -  2: Trung tính (nhận xét trung tính về khía cạnh đó)

Công thức: Am = (Po - Pe) / (1 - Pe)
  - Po: xác suất đồng thuận quan sát (observed agreement)
  - Pe: xác suất đồng thuận ngẫu nhiên (expected agreement by chance)
  - Am = 0: đồng thuận bằng ngẫu nhiên
  - Am = 1: đồng thuận hoàn toàn

Thang diễn giải (Landis & Koch, 1977):
  < 0.00: Không đồng thuận
  0.00–0.20: Rất yếu (Slight)
  0.20–0.40: Yếu (Fair)
  0.40–0.60: Trung bình (Moderate)
  0.60–0.80: Khá (Substantial)
  0.80–1.00: Gần hoàn toàn (Almost Perfect)
"""

import pandas as pd
import numpy as np
from collections import Counter
import os

# 8 nhãn gán trong bộ dữ liệu
LABEL_COLS = ['Price', 'Shipping', 'Outlook', 'Quality', 'Size', 'Shop_Service', 'General', 'Others']


def load_data(data_dir="."):
    """Đọc và gộp 3 file Excel (Train, Test, Validate) thành 1 DataFrame."""
    # Tìm file trong data_dir hoặc data_dir/Data
    search_dirs = [data_dir, os.path.join(data_dir, "Data")]
    dfs = []
    for f in ['Shoes_Train_Data.xlsx', 'Shoes_Test_Data.xlsx', 'Shoes_Validate_Data.xlsx']:
        for d in search_dirs:
            path = os.path.join(d, f)
            if os.path.exists(path):
                dfs.append(pd.read_excel(path))
                break
    if not dfs:
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu trong: {search_dirs}")
    return pd.concat(dfs, ignore_index=True)


def compute_am(values):
    """
    Tính Am cho một nhãn (single-label agreement).
    
    Bước 1: Loại bỏ giá trị -1 (không liên quan)
    Bước 2: Tính Po = sum(p_j^2) — xác suất 2 lần gán ngẫu nhiên trùng nhau
    Bước 3: Tính Pe = 1/m — xác suất trùng nhau nếu gán ngẫu nhiên đều
    Bước 4: Am = (Po - Pe) / (1 - Pe)
    """
    relevant = values[values != -1]
    if len(relevant) < 2:
        return np.nan
    m = len(set(relevant))  # số category khác nhau
    if m < 2:
        return np.nan  # chỉ 1 category thì không thể đo đồng thuận
    n = len(relevant)
    counts = Counter(relevant)
    Po = sum((c / n) ** 2 for c in counts.values())  # Po = Σ(p_j²)
    Pe = 1 / m  # Pe = 1/m (chance agreement)
    return (Po - Pe) / (1 - Pe)


def compute_am_pairwise(df, label_cols):
    """
    Tính Am pairwise trung bình giữa tất cả các cặp nhãn.
    
    Coi mỗi nhãn là một annotator, mỗi comment là một item.
    Chỉ tính trên các comment mà cả 2 nhãn đều có dữ liệu (≠ -1).
    Am trung bình cho biết các khía cạnh gán nhãn có nhất quán với nhau hay không.
    """
    ams = []
    for i, c1 in enumerate(label_cols):
        for c2 in label_cols[i+1:]:  # duyệt các cặp không trùng
            # Lọc comment có cả 2 nhãn đều liên quan
            mask = (df[c1] != -1) & (df[c2] != -1)
            sub = df[mask]
            if len(sub) == 0:
                continue
            v1, v2 = sub[c1].values, sub[c2].values
            # Po: tỷ lệ 2 nhãn gán cùng giá trị trên cùng 1 comment
            agree = np.sum(v1 == v2)
            Po = agree / len(sub)
            # Pe: xác suất trùng nhau nếu 2 nhãn gán ngẫu nhiên độc lập
            ct1, ct2 = Counter(v1), Counter(v2)
            cats = set(v1) | set(v2)
            n = len(sub)
            Pe = sum((ct1.get(c, 0) / n) * (ct2.get(c, 0) / n) for c in cats)
            if Pe < 1:
                ams.append((Po - Pe) / (1 - Pe))
    return np.mean(ams) if ams else np.nan


def interpret(am):
    """Diễn giải giá trị Am theo thang Landis & Koch (1977)."""
    if np.isnan(am): return "N/A"
    for threshold, label in [(0, "Không đồng thuận"), (0.2, "Rất yếu"), (0.4, "Yếu"),
                             (0.6, "Trung bình"), (0.8, "Khá"), (1.01, "Gần hoàn toàn")]:
        if am < threshold:
            return label
    return "Gần hoàn toàn"


def bar(am, w=25):
    """Tạo thanh trực quan: █ = đồng thuận, ░ = không đồng thuận."""
    if np.isnan(am): return "░" * w
    f = max(0, min(w, int(round(am * w))))
    return "█" * f + "░" * (w - f)


def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    df = load_data(data_dir)

    # Tính Am cho từng nhãn riêng lẻ
    per_label = {col: compute_am(df[col].values) for col in LABEL_COLS}
    # Tính Am tổng thể (trung bình pairwise giữa các cặp nhãn)
    overall = compute_am_pairwise(df, LABEL_COLS)

    # ── Thông tin dữ liệu ──
    print(f"\n  PHÉP ĐO ĐỒNG THUẬN Am — Comment Shopee về giày\n")
    print(f"  Tổng: {len(df):,} comment, mỗi comment được gán 8 nhãn.")
    print(f"  Giá trị gán: -1 = Không liên quan  |  0 = Tiêu cực  |  1 = Tích cực  |  2 = Trung tính")
    print()
    print(f"  {'Nhãn':<14} {'Không liên quan':>15} {'Có nhãn':>8} {'Tỷ lệ có':>9}   Tiêu cực  Tích cực  Trung tính")
    print(f"  {'─'*14} {'─'*15} {'─'*8} {'─'*9}   {'─'*9}  {'─'*8}  {'─'*11}")
    total_neg, total_pos, total_neu = 0, 0, 0
    for col in LABEL_COLS:
        vals = df[col].values
        n_total = len(vals)
        n_neg1 = np.sum(vals == -1)
        relevant = vals[vals != -1]
        n_rel = len(relevant)
        pct = f"{n_rel/n_total*100:.1f}%"
        ct = Counter(relevant)
        v0, v1, v2 = ct.get(0, 0), ct.get(1, 0), ct.get(2, 0)
        total_neg += v0
        total_pos += v1
        total_neu += v2
        print(f"  {col:<14} {n_neg1:>15,} {n_rel:>8,} {pct:>9}   {v0:>9,}  {v1:>8,}  {v2:>11,}")
    total_all = total_neg + total_pos + total_neu
    print(f"  {'─'*14} {'─'*15} {'─'*8} {'─'*9}   {'─'*9}  {'─'*8}  {'─'*11}")
    print(f"  {'Tổng cộng':<14} {'':>15} {total_all:>8,} {'':>9}   {total_neg:>9,}  {total_pos:>8,}  {total_neu:>11,}")

    # ── Kết quả Am ──
    print(f"\n  {'Nhãn':<14} {'Am':>7}  {'Bar':<25}  Diễn giải")
    print(f"  {'─'*14} {'─'*7}  {'─'*25}  {'─'*12}")
    for col in LABEL_COLS:
        am = per_label[col]
        print(f"  {col:<14} {am:>7.4f}  {bar(am)}  {interpret(am)}" if not np.isnan(am)
              else f"  {col:<14} {'N/A':>7}  {bar(am)}  {interpret(am)}")

    print(f"\n  Am tổng thể: {overall:.4f} ({interpret(overall)})")
    print(f"  → Dữ liệu {'KHÔNG ĐỒNG THUẬN CAO' if overall < 0.4 else 'CÓ ĐỒNG THUẬN CAO'}")
    print(f"\n  Ghi chú: Cột 'Others' chỉ có giá trị -1 và 2 (trung tính),")
    print(f"  nên không thể tính Am (chỉ 1 category sau khi loại -1).\n")


if __name__ == "__main__":
    main()