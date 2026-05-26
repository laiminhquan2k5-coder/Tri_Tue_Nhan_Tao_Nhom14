# =============================================================================
#  ViSoBERT Tokenization Pipeline
#  Dữ liệu: Shoes_Train_Data.xlsx (10 dòng được chọn)
#
#  Yêu cầu:
#    pip install transformers torch pandas openpyxl
#
#  Chạy:
#    python visobert_tokenization.py
# =============================================================================

import pandas as pd
import torch
from transformers import AutoTokenizer

# =============================================================================
# 1. LOAD DỮ LIỆU — 10 dòng đa dạng từ Shoes_Train_Data.xlsx
# =============================================================================

# Đọc file Excel (đổi đường dẫn nếu cần)
df = pd.read_excel("Shoes_Train_Data.xlsx")

# Chọn 10 chỉ số đại diện: ngắn, trung bình, dài; nhãn đa dạng
SELECTED_INDICES = [0, 1, 2, 3, 5, 6, 7, 10, 14, 25]
sample_df = df.iloc[SELECTED_INDICES][["Review"]].reset_index(drop=True)
texts = sample_df["Review"].astype(str).tolist()

print("=" * 70)
print("BƯỚC 0 — 10 DÒNG DỮ LIỆU ĐƯỢC CHỌN")
print("=" * 70)
for i, t in enumerate(texts):
    preview = t[:80] + ("..." if len(t) > 80 else "")
    print(f"  [{i+1:02d}] ({len(t.split())} từ) {preview}")

# =============================================================================
# 2. KHỞI TẠO AUTOTOKENIZER CỦA VISOBERT
# =============================================================================
# ViSoBERT được công bố bởi UIT-NLP Lab:
#   https://huggingface.co/uitnlp/visobert
# Sử dụng WordPiece tokenizer giống BERT nhưng được huấn luyện trên văn bản
# tiếng Việt mạng xã hội (Facebook, Twitter, Tiki reviews...).

MODEL_NAME = "uitnlp/visobert"
MAX_LENGTH = 64  # padding tất cả về cùng độ dài

print("\n" + "=" * 70)
print("BƯỚC 1 — KHỞI TẠO AUTOTOKENIZER")
print("=" * 70)
print(f"  Model : {MODEL_NAME}")
print(f"  Max length (sau padding): {MAX_LENGTH}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"  Vocab size : {tokenizer.vocab_size:,}")
print(f"  [CLS] token: '{tokenizer.cls_token}'  (ID = {tokenizer.cls_token_id})")
print(f"  [SEP] token: '{tokenizer.sep_token}'  (ID = {tokenizer.sep_token_id})")
print(f"  [PAD] token: '{tokenizer.pad_token}'  (ID = {tokenizer.pad_token_id})")
print(f"  [UNK] token: '{tokenizer.unk_token}'  (ID = {tokenizer.unk_token_id})")

# =============================================================================
# 3. TOKENIZATION TỪNG CÂU — hiển thị danh sách token
# =============================================================================

print("\n" + "=" * 70)
print("BƯỚC 2 — TOKENIZATION TỪNG CÂU (chưa padding)")
print("=" * 70)

for i, text in enumerate(texts):
    tokens = tokenizer.tokenize(text)
    tokens_with_special = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    print(f"\n  Review {i+1:02d}: {text[:65]}{'...' if len(text) > 65 else ''}")
    print(f"  Số tokens (kèm [CLS]/[SEP]): {len(tokens_with_special)}")
    print(f"  Tokens: {tokens_with_special}")

# =============================================================================
# 4. ENCODE TOÀN BỘ — PADDING + ATTENTION MASK + TENSOR
# =============================================================================
# Tham số:
#   padding=True      → pad ngắn hơn bằng [PAD] token
#   truncation=True   → cắt dài hơn MAX_LENGTH
#   max_length=64     → độ dài đồng nhất
#   return_tensors="pt" → trả về PyTorch tensor

print("\n" + "=" * 70)
print("BƯỚC 3 — ENCODE BATCH: PADDING + TRUNCATION + TENSORS")
print("=" * 70)

encoded = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",   # "pt" = PyTorch, dùng "tf" cho TensorFlow
)

input_ids      = encoded["input_ids"]        # shape [10, 64]
attention_mask = encoded["attention_mask"]   # shape [10, 64]

print(f"  input_ids.shape      : {tuple(input_ids.shape)}")
print(f"  attention_mask.shape : {tuple(attention_mask.shape)}")
print(f"  dtype                : {input_ids.dtype}")

# =============================================================================
# 5. HIỂN THỊ TOKEN IDS
# =============================================================================

print("\n" + "=" * 70)
print("BƯỚC 4 — TOKEN IDs (mỗi dòng = 1 review, 64 giá trị)")
print("=" * 70)

for i in range(len(texts)):
    ids = input_ids[i].tolist()
    n_real = int(attention_mask[i].sum().item())
    print(f"\n  Review {i+1:02d} | {n_real} token thực + {MAX_LENGTH - n_real} [PAD]")
    # Hiển thị 20 giá trị đầu để gọn
    preview_ids = ids[:20]
    suffix = f" ... ({MAX_LENGTH - 20} nữa)" if MAX_LENGTH > 20 else ""
    print(f"  IDs: {preview_ids}{suffix}")

# =============================================================================
# 6. HIỂN THỊ ATTENTION MASK
# =============================================================================

print("\n" + "=" * 70)
print("BƯỚC 5 — ATTENTION MASK (1 = token thực, 0 = padding)")
print("=" * 70)

for i in range(len(texts)):
    mask   = attention_mask[i].tolist()
    n_real = int(attention_mask[i].sum().item())
    n_pad  = MAX_LENGTH - n_real

    # Trực quan hóa dạng chuỗi
    visual = "█" * n_real + "░" * n_pad
    print(f"  R{i+1:02d} [{visual}] {n_real:2d} thực / {n_pad:2d} pad")

# =============================================================================
# 7. BẢNG ĐỐI CHIẾU TOKEN – ID – MASK (10 tokens đầu mỗi review)
# =============================================================================

print("\n" + "=" * 70)
print("BƯỚC 6 — BẢNG ĐỐI CHIẾU: TOKEN | ID | MASK (10 vị trí đầu)")
print("=" * 70)

for i in range(len(texts)):
    ids   = input_ids[i].tolist()
    masks = attention_mask[i].tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)

    print(f"\n  Review {i+1:02d}: {texts[i][:60]}{'...' if len(texts[i]) > 60 else ''}")
    print(f"  {'Pos':>4}  {'Token':<18}  {'ID':>6}  {'Mask'}")
    print(f"  {'-'*4}  {'-'*18}  {'-'*6}  {'-'*4}")
    for pos in range(min(10, MAX_LENGTH)):
        tok  = tokens[pos]
        tid  = ids[pos]
        mval = masks[pos]
        marker = "◀ [CLS]" if pos == 0 else ("◀ [SEP]" if tok == "[SEP]" else ("◀ [PAD]" if tok == "[PAD]" else ""))
        print(f"  {pos:>4}  {tok:<18}  {tid:>6}  {mval}    {marker}")

# =============================================================================
# 8. GIẢI THÍCH LUỒNG DỮ LIỆU VÀO VISOBERT
# =============================================================================

print("\n" + "=" * 70)
print("BƯỚC 7 — LUỒNG DỮ LIỆU VÀO MÔ HÌNH VISOBERT")
print("=" * 70)

explanation = """
  Sau khi tokenization, 3 tensor được truyền vào ViSoBERT:

  ┌─────────────────────────────────────────────────────────────────┐
  │  1. input_ids      [10, 64]                                     │
  │     Mỗi token được ánh xạ sang số nguyên (chỉ số trong vocab). │
  │     [CLS]=0, [PAD]=1, [SEP]=2, token thường ∈ [200, 30000].   │
  │                                                                 │
  │  2. attention_mask [10, 64]                                     │
  │     1 = model chú ý token này (token thực)                     │
  │     0 = model bỏ qua (padding) — tránh học nhiễu              │
  │                                                                 │
  │  3. token_type_ids [10, 64]  (tùy chọn, mặc định toàn 0)      │
  │     Phân biệt câu A và câu B trong bài NSP — không cần        │
  │     thiết khi chỉ có 1 câu đầu vào.                            │
  └─────────────────────────────────────────────────────────────────┘

  Luồng xử lý bên trong ViSoBERT (12 tầng Transformer):

    input_ids
        │
        ▼
  [Embedding Layer]  → token emb (768d) + position emb + segment emb
        │
        ▼
  [Transformer × 12] ← attention_mask kiểm soát self-attention
        │
        ▼
  [Hidden States: 10 × 64 × 768]
        │
        ▼
  Lấy vector [CLS] ở vị trí 0
  → shape [10, 768]
        │
        ▼
  [Linear + Softmax]
        │
        ▼
  Dự đoán nhãn cho 8 khía cạnh:
    Price | Shipping | Outlook | Quality | Size | Shop_Service | General | Others
"""
print(explanation)

# =============================================================================
# 9. VÍ DỤ TRUYỀN VÀO MÔ HÌNH (comment ra nếu chưa có GPU/model weights)
# =============================================================================
# Bỏ comment khối dưới nếu muốn chạy thực sự với ViSoBERT:

# from transformers import AutoModel
#
# model = AutoModel.from_pretrained(MODEL_NAME)
# model.eval()
#
# with torch.no_grad():
#     outputs = model(
#         input_ids=input_ids,
#         attention_mask=attention_mask,
#     )
#
# # Lấy vector [CLS] — đại diện toàn câu
# cls_embeddings = outputs.last_hidden_state[:, 0, :]
# print(f"\n  CLS embeddings shape: {tuple(cls_embeddings.shape)}")
# # → [10, 768]  —  mỗi review được biểu diễn bằng vector 768 chiều

print("=" * 70)
print("HOÀN THÀNH pipeline tokenization ViSoBERT.")
print("=" * 70)
