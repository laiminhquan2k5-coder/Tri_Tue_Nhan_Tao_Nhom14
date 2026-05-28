# -*- coding: utf-8 -*-
"""
PIPELINE TIỀN XỬ LÝ DỮ LIỆU — 2.3 Tiền xử lý dữ liệu
  2.3.1 Lowercase
  2.3.2 Remove punctuation
  2.3.3 Remove URL
  2.3.4 Remove emoji
  2.3.5 Chuẩn hóa từ viết tắt và teencode
  2.3.6 Tokenization bằng ViSoBERT
  2.3.7 Padding sequence
  2.3.8 Attention Mask
"""

import os
import sys
import pandas as pd

# Thêm thư mục hiện tại vào sys.path để import module cùng cấp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from xuly_rac import remove_gibberish
from tienxuly_vanban import (
    process_lowercase,            # 2.3.1
    process_remove_punctuation,   # 2.3.2
    process_remove_url,           # 2.3.3
    process_remove_emoji,         # 2.3.4
    process_normalize_teencode,   # 2.3.5
    process_all_text_preprocessing,
)
from visobert_tokenize import (
    process_tokenization,         # 2.3.6
    process_padding,              # 2.3.7
    process_attention_mask,       # 2.3.8
    process_all_tokenization,
)

# ── Cấu hình đường dẫn ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Data")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

FILES = {
    "Train":    os.path.join(DATA_DIR, "Shoes_Train_Data.xlsx"),
    "Test":     os.path.join(DATA_DIR, "Shoes_Test_Data.xlsx"),
    "Validate": os.path.join(DATA_DIR, "Shoes_Validate_Data.xlsx"),
}


def run_pipeline():
    """Chạy toàn bộ 8 bước tiền xử lý và xuất file CSV."""

    print("=" * 70)
    print("  2.3 PIPELINE TIỀN XỬ LÝ DỮ LIỆU GIÀY SHOPEE")
    print("=" * 70)

    # Lưu thông tin tổng kết
    summary = {}

    for name, path in FILES.items():
        if not os.path.exists(path):
            print(f"  ❌ Không tìm thấy file: {path}")
            continue

        print(f"\n{'━' * 70}")
        print(f"  📂 Tập: {name}")
        print(f"  📄 File: {path}")
        print(f"{'━' * 70}")

        # ── Đọc dữ liệu gốc ─────────────────────────────────────────
        df_raw = pd.read_excel(path)
        raw_rows, raw_cols = len(df_raw), len(df_raw.columns)
        print(f"\n  Đọc dữ liệu gốc: {raw_rows:,} dòng, {raw_cols} cột")
        print(f"  Các cột: {list(df_raw.columns)}")

        # ── Xóa dòng rác (tiền bước) ────────────────────────────────
        df, n_removed = remove_gibberish(df_raw)
        print(f"\n  Xóa dòng rác: {n_removed} dòng → còn {len(df):,} dòng")

        # ════════════════════════════════════════════════════════════
        # 2.3.1 LOWERCASE
        # ════════════════════════════════════════════════════════════
        df = process_lowercase(df, col="Review")
        print(f"\n  [2.3.1] Lowercase")
        print(f"          + Tạo cột 'Review_Lowercase'")
        print(f"          Ví dụ: '{df['Review'].iloc[0][:40]}' → '{df['Review_Lowercase'].iloc[0][:40]}'")

        # ════════════════════════════════════════════════════════════
        # 2.3.2 REMOVE PUNCTUATION
        # ════════════════════════════════════════════════════════════
        df = process_remove_punctuation(df, col="Review_Lowercase")
        print(f"\n  [2.3.2] Remove punctuation")
        print(f"          + Tạo cột 'Review_NoPunct'")

        # ════════════════════════════════════════════════════════════
        # 2.3.3 REMOVE URL
        # ════════════════════════════════════════════════════════════
        df = process_remove_url(df, col="Review_NoPunct")
        print(f"\n  [2.3.3] Remove URL")
        print(f"          + Tạo cột 'Review_NoURL'")

        # ════════════════════════════════════════════════════════════
        # 2.3.4 REMOVE EMOJI
        # ════════════════════════════════════════════════════════════
        df = process_remove_emoji(df, col="Review_NoURL")
        print(f"\n  [2.3.4] Remove emoji")
        print(f"          + Tạo cột 'Review_NoEmoji'")

        # ════════════════════════════════════════════════════════════
        # 2.3.5 CHUẨN HÓA TỪ VIẾT TẮT VÀ TEENCODE
        # ════════════════════════════════════════════════════════════
        df = process_normalize_teencode(df, col="Review_NoEmoji")
        print(f"\n  [2.3.5] Chuẩn hóa từ viết tắt và teencode")
        print(f"          + Tạo cột 'Review_Cleaned' (kết quả 5 bước)")

        # ════════════════════════════════════════════════════════════
        # 2.3.6 TOKENIZATION BẰNG VISOBERT
        # ════════════════════════════════════════════════════════════
        try:
            df = process_tokenization(df, review_col="Review_Cleaned")
            print(f"\n  [2.3.6] Tokenization bằng ViSoBERT")
            print(f"          + Tạo cột 'tokens'")
            # Hiển thị ví dụ
            sample_tokens = df["tokens"].iloc[0]
            print(f"          Ví dụ: {sample_tokens[:8]}...")
        except Exception as e:
            print(f"\n  [2.3.6] Tokenization: ⚠️ Bỏ qua ({e})")

        # ════════════════════════════════════════════════════════════
        # 2.3.7 PADDING SEQUENCE
        # ════════════════════════════════════════════════════════════
        try:
            df = process_padding(df, review_col="Review_Cleaned")
            print(f"\n  [2.3.7] Padding sequence")
            print(f"          + Tạo cột 'input_ids' (pad về độ dài 64)")
        except Exception as e:
            print(f"\n  [2.3.7] Padding: ⚠️ Bỏ qua ({e})")

        # ════════════════════════════════════════════════════════════
        # 2.3.8 ATTENTION MASK
        # ════════════════════════════════════════════════════════════
        try:
            df = process_attention_mask(df, review_col="Review_Cleaned")
            print(f"\n  [2.3.8] Attention Mask")
            print(f"          + Tạo cột 'attention_mask' (1=token thực, 0=padding)")
        except Exception as e:
            print(f"\n  [2.3.8] Attention Mask: ⚠️ Bỏ qua ({e})")

        # ── Xuất file CSV (chỉ cột cần thiết) ────────────────────────
        output_csv = os.path.join(OUTPUT_DIR, f"Shoes_{name}_Preprocessed.csv")

        # File dữ liệu chính: chỉ giữ cột gốc + Review_Cleaned + tokenization
        main_cols = [c for c in df.columns if c not in [
            "Review_Lowercase", "Review_NoPunct", "Review_NoURL", "Review_NoEmoji"
        ]]
        df_export = df[main_cols].copy()
        for col in ["tokens", "input_ids", "attention_mask"]:
            if col in df_export.columns:
                df_export[col] = df_export[col].apply(
                    lambda x: str(x) if isinstance(x, list) else x
                )

        df_export.to_csv(output_csv, index=False, encoding="utf-8-sig")

        print(f"\n  📊 Xuất file CSV:")
        print(f"     {output_csv}")
        print(f"     → {len(df_export):,} dòng, {len(df_export.columns)} cột")
        print(f"     → Các cột: {list(df_export.columns)}")

        # Lưu thông tin cho tổng kết
        summary[name] = {
            "raw_rows": raw_rows,
            "raw_cols": raw_cols,
            "removed": n_removed,
            "out_rows": len(df_export),
            "out_cols": len(df_export.columns),
        }

    # ── Tổng kết ────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  TỔNG KẾT KẾT QUẢ TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 70)

    # Tính tổng
    total_raw = sum(v["raw_rows"] for v in summary.values())
    total_removed = sum(v["removed"] for v in summary.values())
    total_out = sum(v["out_rows"] for v in summary.values())
    total_pct = (total_removed / total_raw * 100) if total_raw else 0

    # Bảng tổng quan
    print()
    print("  ┌───────────┬──────────────────┬──────────┬───────────┬──────────────────┐")
    print("  │   Tập     │   Dữ liệu gốc    │  Xóa rác │  Tỷ lệ    │  Sau xử lý       │")
    print("  ├───────────┼──────────────────┼──────────┼───────────┼──────────────────┤")
    for name, info in summary.items():
        before  = f"{info['raw_rows']:,} dòng × {info['raw_cols']} cột"
        after   = f"{info['out_rows']:,} dòng × {info['out_cols']} cột"
        removed = f"{info['removed']:,} dòng"
        pct_i   = (info['removed'] / info['raw_rows'] * 100) if info['raw_rows'] else 0
        print(f"  │ {name:<9} │ {before:<16} │ {removed:<8} │ {pct_i:<8.1f}% │ {after:<16} │")
    print("  ├───────────┼──────────────────┼──────────┼───────────┼──────────────────┤")
    print(f"  │ {'Tổng':<9} │ {total_raw:,} dòng × 9 cột  │ {total_removed:,} dòng │ {total_pct:<7.1f}% │ {total_out:,} dòng × 13 cột │")
    print("  └───────────┴──────────────────┴──────────┴───────────┴──────────────────┘")

    # Tóm tắt ngắn
    print()
    print("  📌 Pipeline: 9 cột gốc → +4 cột mới (Review_Cleaned, tokens, input_ids, attention_mask)")
    print(f"  📌 Xóa rác: {total_removed:,}/{total_raw:,} dòng ({total_pct:.1f}%)")
    print(f"  📌 Kết quả: {total_out:,} dòng × 13 cột  |  ViSoBERT | max_length=64")

    print()
    print("=" * 70)
    print("  ✅ HOÀN THÀNH PIPELINE TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()