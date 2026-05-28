# -*- coding: utf-8 -*-
"""
Module: Tiền xử lý văn bản — 5 bước đầu (2.3.1 → 2.3.5)

  2.3.1 Lowercase
  2.3.2 Remove punctuation
  2.3.3 Remove URL
  2.3.4 Remove emoji
  2.3.5 Chuẩn hóa từ viết tắt và teencode

Đầu vào:  DataFrame gốc (có cột Review)
Đầu ra:   DataFrame với cột Review_Cleaned (đã qua 5 bước)
"""

import re
import string
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════
# 2.3.1 LOWERCASE
# ═══════════════════════════════════════════════════════════════════════

def lowercase(text):
    """Chuyển văn bản về chữ thường."""
    if not isinstance(text, str):
        return ""
    return text.lower()


def process_lowercase(df, col="Review"):
    """Áp dụng lowercase lên cột chỉ định."""
    df = df.copy()
    df["Review_Lowercase"] = df[col].astype(str).apply(lowercase)
    return df


# ═══════════════════════════════════════════════════════════════════════
# 2.3.2 REMOVE PUNCTUATION
# ═══════════════════════════════════════════════════════════════════════

PUNCT_TABLE = str.maketrans('', '', string.punctuation)


def remove_punctuation(text):
    """Xóa dấu câu (punctuation) khỏi văn bản."""
    if not isinstance(text, str):
        return ""
    return text.translate(PUNCT_TABLE)


def process_remove_punctuation(df, col="Review_Lowercase"):
    """Áp dụng remove punctuation lên cột chỉ định."""
    df = df.copy()
    df["Review_NoPunct"] = df[col].astype(str).apply(remove_punctuation)
    return df


# ═══════════════════════════════════════════════════════════════════════
# 2.3.3 REMOVE URL
# ═══════════════════════════════════════════════════════════════════════

URL_PATTERN = r'https?://\S+|www\.\S+'


def remove_url(text):
    """Xóa URL khỏi văn bản."""
    if not isinstance(text, str):
        return ""
    return re.sub(URL_PATTERN, '', text)


def process_remove_url(df, col="Review_NoPunct"):
    """Áp dụng remove URL lên cột chỉ định."""
    df = df.copy()
    df["Review_NoURL"] = df[col].astype(str).apply(remove_url)
    return df


# ═══════════════════════════════════════════════════════════════════════
# 2.3.4 REMOVE EMOJI
# ═══════════════════════════════════════════════════════════════════════

def remove_emoji(text):
    """
    Xóa emoji, icon, ký tự đặc biệt.
    Chỉ giữ lại: chữ cái, số, khoảng trắng, dấu chấm, dấu phẩy.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^\w\s,.]|_", "", text, flags=re.UNICODE)
    # Chuẩn hóa khoảng trắng
    text = re.sub(r"\s+", " ", text).strip()
    return text


def process_remove_emoji(df, col="Review_NoURL"):
    """Áp dụng remove emoji lên cột chỉ định."""
    df = df.copy()
    df["Review_NoEmoji"] = df[col].astype(str).apply(remove_emoji)
    return df


# ═══════════════════════════════════════════════════════════════════════
# 2.3.5 CHUẨN HÓA TỪ VIẾT TẮT VÀ TEENCODE
# ═══════════════════════════════════════════════════════════════════════

TEENCODE_DICT = {
    r"\bsp\b|\bshp\b": "sản phẩm",
    r"\bshb\b|\bshop\b": "cửa hàng",
    r"\bk\b|\bko\b|\bkhg\b|\bkh\b": "không",
    r"\bmn\b|\bmng\b": "mọi người",
    r"\bsz\b": "size",
    r"\bđc\b|\bdc\b": "được",
    r"\bokie\b|\bok\b|\boke\b": "ổn",
    r"\bnhma\b": "nhưng mà",
    r"\bvs\b": "với",
    r"\bđg\b": "đang",
    r"\bbth\b|\bbt\b": "bình thường",
    r"\bqá\b|\bqa\b": "quá",
    r"\btks\b|\bthanks\b": "cảm ơn",
    r"\bgiầy\b": "giày",
    r"\bđt\b": "đặt",
}

# Ký tự tiếng Việt dùng cho regex rút gọn lặp
VIET_CHARS = (
    "a-zàáạảãâầấậẩẫăằắặẳẵ"
    "èéẹẻẽêềếệểễ"
    "ìíịỉĩ"
    "òóọỏõôồốộổỗơờớợởỡ"
    "ùúụủũưừứựửữ"
    "ỳýỵỷỹđ"
)


def normalize_teencode(text):
    """
    Chuẩn hóa từ viết tắt, teencode TMĐT Gen Z.
    Rút gọn ký tự lặp kéo dài (đẹpppp → đẹp).
    """
    if not isinstance(text, str):
        return ""

    # 1. Thay thế teencode
    for pattern, replacement in TEENCODE_DICT.items():
        text = re.sub(pattern, replacement, text)

    # 2. Rút gọn ký tự lặp kéo dài
    text = re.sub(r"([" + VIET_CHARS + r"])\1+", r"\1", text)

    # 3. Chuẩn hóa khoảng trắng
    text = re.sub(r"\s+", " ", text).strip()

    return text


def process_normalize_teencode(df, col="Review_NoEmoji"):
    """Áp dụng chuẩn hóa teencode lên cột chỉ định."""
    df = df.copy()
    df["Review_Cleaned"] = df[col].astype(str).apply(normalize_teencode)
    return df


# ═══════════════════════════════════════════════════════════════════════
# HÀM CHÍNH — Chạy tất cả 5 bước liên tiếp
# ═══════════════════════════════════════════════════════════════════════

def process_all_text_preprocessing(df, review_col="Review"):
    """
    Chạy liên tiếp 5 bước tiền xử lý văn bản:
      2.3.1 Lowercase
      2.3.2 Remove punctuation
      2.3.3 Remove URL
      2.3.4 Remove emoji
      2.3.5 Chuẩn hóa teencode
    
    Args:
        df: DataFrame gốc
        review_col: tên cột review gốc
    Returns:
        df: DataFrame với các cột trung gian + cột 'Review_Cleaned' (kết quả cuối)
    """
    df = process_lowercase(df, col=review_col)           # 2.3.1
    df = process_remove_punctuation(df)                   # 2.3.2
    df = process_remove_url(df)                            # 2.3.3
    df = process_remove_emoji(df)                          # 2.3.4
    df = process_normalize_teencode(df)                    # 2.3.5
    return df