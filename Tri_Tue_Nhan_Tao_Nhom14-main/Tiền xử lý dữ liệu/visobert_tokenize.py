# -*- coding: utf-8 -*-
"""
Module: ViSoBERT Tokenization — 3 bước cuối (2.3.6 → 2.3.8)

  2.3.6 Tokenization bằng ViSoBERT
  2.3.7 Padding sequence
  2.3.8 Attention Mask

Đầu vào:  DataFrame đã qua 5 bước (có cột Review_Cleaned)
Đầu ra:   DataFrame với cột input_ids, attention_mask
"""

import pandas as pd
from transformers import AutoTokenizer

MODEL_NAME = "uitnlp/visobert"
MAX_LENGTH = 64

# Cache tokenizer toàn cục để không load lại nhiều lần
_tokenizer = None


# ═══════════════════════════════════════════════════════════════════════
# 2.3.6 TOKENIZATION BẰNG VISOBERT
# ═══════════════════════════════════════════════════════════════════════

def get_tokenizer():
    """Lấy (hoặc khởi tạo) ViSoBERT tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tokenizer


def tokenize_single_texts(texts):
    """
    2.3.6: Tokenize danh sách văn bản bằng ViSoBERT (chưa padding).
    
    Args:
        texts: list[str] — danh sách review đã tiền xử lý
    Returns:
        list[list[str]] — danh sách tokens cho mỗi review
    """
    tokenizer = get_tokenizer()
    all_tokens = []
    for text in texts:
        tokens = tokenizer.tokenize(str(text))
        all_tokens.append(tokens)
    return all_tokens


def process_tokenization(df, review_col="Review_Cleaned"):
    """
    2.3.6: Tokenize cột review trong DataFrame.
    Tạo cột 'tokens' chứa danh sách token cho mỗi review.
    """
    df = df.copy()
    texts = df[review_col].astype(str).tolist()
    df["tokens"] = tokenize_single_texts(texts)
    return df


# ═══════════════════════════════════════════════════════════════════════
# 2.3.7 PADDING SEQUENCE
# ═══════════════════════════════════════════════════════════════════════

def pad_sequences(texts, max_length=MAX_LENGTH):
    """
    2.3.7: Encode + Pad tất cả chuỗi về cùng độ dài max_length.
    
    Args:
        texts: list[str] — danh sách review
        max_length: int — độ dài đồng nhất sau padding
    Returns:
        dict: input_ids (list[list[int]]), attention_mask (list[list[int]])
    """
    tokenizer = get_tokenizer()
    encoded = tokenizer(
        texts,
        padding="max_length",       # pad tất cả về max_length
        truncation=True,             # cắt dài hơn max_length
        max_length=max_length,
        return_tensors=None,         # trả về list Python
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


def process_padding(df, review_col="Review_Cleaned", max_length=MAX_LENGTH):
    """
    2.3.7: Padding sequence — pad tất cả review về cùng độ dài.
    Tạo cột 'input_ids' chứa danh sách ID đã pad.
    """
    df = df.copy()
    texts = df[review_col].astype(str).tolist()
    encodings = pad_sequences(texts, max_length=max_length)
    df["input_ids"] = encodings["input_ids"]
    df["attention_mask"] = encodings["attention_mask"]
    return df


# ═══════════════════════════════════════════════════════════════════════
# 2.3.8 ATTENTION MASK
# ═══════════════════════════════════════════════════════════════════════

def process_attention_mask(df, review_col="Review_Cleaned", max_length=MAX_LENGTH):
    """
    2.3.8: Attention Mask — tạo mask cho model bỏ qua padding tokens.
    1 = token thực (model chú ý), 0 = padding (model bỏ qua).
    
    Lưu ý: attention_mask được tạo cùng lúc với padding (2.3.7),
    hàm này tách riêng để minh họa rõ bước trong tài liệu.
    """
    df = df.copy()
    texts = df[review_col].astype(str).tolist()
    encodings = pad_sequences(texts, max_length=max_length)
    df["input_ids"] = encodings["input_ids"]
    df["attention_mask"] = encodings["attention_mask"]
    return df


# ═══════════════════════════════════════════════════════════════════════
# HÀM CHÍNH — Chạy 3 bước tokenization liên tiếp
# ═══════════════════════════════════════════════════════════════════════

def process_all_tokenization(df, review_col="Review_Cleaned", max_length=MAX_LENGTH):
    """
    Chạy liên tiếp 3 bước tokenization:
      2.3.6 Tokenization bằng ViSoBERT
      2.3.7 Padding sequence
      2.3.8 Attention Mask
    
    Args:
        df: DataFrame đã qua 5 bước tiền xử lý
        review_col: tên cột review đã làm sạch
        max_length: độ dài padding/truncation
    Returns:
        df: DataFrame với cột 'tokens', 'input_ids', 'attention_mask'
    """
    df = process_tokenization(df, review_col=review_col)       # 2.3.6
    df = process_padding(df, review_col=review_col,           # 2.3.7
                         max_length=max_length)
    # 2.3.8 Attention mask đã được tạo cùng padding
    return df