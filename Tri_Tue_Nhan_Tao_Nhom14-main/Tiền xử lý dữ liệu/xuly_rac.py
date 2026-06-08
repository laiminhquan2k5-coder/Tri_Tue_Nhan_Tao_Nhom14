# -*- coding: utf-8 -*-
"""
Module: Xử lý dữ liệu rác — Xóa dòng gibberish

Đầu vào:  file Excel gốc (Shoes_Train/Test/Validate_Data.xlsx)
Đầu ra:   DataFrame đã lọc rác
"""

import re
import pandas as pd


# ── Blacklist ký tự gõ càn ──────────────────────────────────────────
GIBBERISH_BLACKLIST = [
    r"sidbd", r"síbd", r"gti", r"dbd", r"sbd",
    r"hhd", r"shs", r"jxj", r"hjh",
    r"([a-z])\1{4,}",  # Spam chữ dài xxxxx, jjjjj
]


def is_strictly_gibberish(text):
    """Trả về True nếu dòng text là rác (gibberish)."""
    if not isinstance(text, str) or not text.strip():
        return True
    text_clean = text.lower().strip()
    for pattern in GIBBERISH_BLACKLIST:
        if re.search(pattern, text_clean):
            return True
    return False


def remove_gibberish(df):
    """
    Xóa các dòng rác khỏi DataFrame.
    
    Args:
        df: DataFrame gốc, phải có cột 'Review'
    Returns:
        df_filtered: DataFrame đã xóa dòng rác
        n_removed: số dòng đã xóa
    """
    mask = ~df["Review"].apply(is_strictly_gibberish)
    df_filtered = df[mask].copy().reset_index(drop=True)
    n_removed = len(df) - len(df_filtered)
    return df_filtered, n_removed