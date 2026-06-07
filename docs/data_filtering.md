# Data Filtering — Vietnamese QA Pipeline

## 1. `src/vietnamese.py`

Module xử lý tiếng Việt: chuẩn hóa, lọc chất lượng, word segmentation, và căn chỉnh answer position sau segment.

---

### 1.1 `normalize_text(text: str) -> str`

Chuẩn hóa Unicode (NFKC) và loại bỏ nhiễu.

| Loại bỏ | Mô tả |
|---------|-------|
| HTML tags | `<br>`, `<div>`, ... |
| URLs | `https://...` |
| Repeated punctuation | `::::`, `||||`, ... |
| Zero-width chars | `\u200b`, `\u200c`, `\u200d`, `\ufeff` |

**Input:**
```python
normalize_text("  Hà Nội là\u200b thủ đô<br>của Việt Nam  ")
```

**Output:**
```
"Hà Nội là thủ đôcủa Việt Nam"
```

---

### 1.2 `is_quality_sample(context, question, answer) -> bool`

Kiểm tra chất lượng một QA sample.

**Reject khi:**
- `len(answer_text) < 2` — quá ngắn
- `len(answer_text) > 150` — quá dài
- `context[answer_start : answer_start+len(answer)] != answer` — answer không khớp context
- `is_impossible` (`answer` rỗng) → **giữ lại** (không reject)

**Input:**
```python
is_quality_sample(
    context="Hà Nội là thủ đô của Việt Nam",
    question="Thủ đô là gì?",
    answer={"text": ["thủ đô"], "answer_start": [15]}
)
# → True  (answer_start=15 trỏ đúng "thủ đô" trong context)

is_quality_sample(
    context="Hà Nội là thủ đô của Việt Nam",
    question="?",
    answer={"text": ["a"], "answer_start": [0]}
)
# → False (answer quá ngắn, len=1)

is_quality_sample(
    context="Hà Nội là thủ đô của Việt Nam",
    question="?",
    answer={"text": [""], "answer_start": [0]}
)
# → True  (impossible question, được giữ lại)
```

---

### 1.3 `segment_texts(texts: list[str]) -> list[str]`

Word segmentation dùng `underthesea.word_tokenize(..., format="text")`.

| Tính chất | Giá trị |
|-----------|---------|
| Output format | `format="text"` — compound nối bằng `_` |
| Thay đổi độ dài text | **Có, ở một số trường hợp** — tokenizer có thể tách số/chữ (ví dụ `số1` → `số 1`) hoặc normalize whitespace |
| API cũ (bug) | `word_tokenize(text)` → `list`, `" ".join()` → text y hệt input (no-op) |

**Input:**
```python
segment_texts(["Tôi là sinh viên", "Hà Nội là thủ đô"])
```

**Output:**
```python
["Tôi là sinh_viên", "Hà_Nội là thủ_đô"]
#                ^                    ^
#           compound           compound
```

> ⚠️ Lưu ý: `format="text"` có thể thêm khoảng cách giữa số và chữ, làm thay đổi `answer_start`. Xem `validate_answer_in_segmented` để fix.

---

### 1.4 `validate_answer_in_segmented(contexts_orig, contexts_seg, answers) -> None`

Cập nhật `answer_text` sau word segmentation (cập nhật `answer_start` nếu vị trí bị dịch do tokenizer thay đổi whitespace).

**Lưu ý:** `segment_texts` (dùng `format="text"`) thay ` ` → `_` trong compound words, là 1:1 (cùng độ dài). `answer_start` thường không đổi — chỉ `answer_text` cần update.

**Cơ chế:**
1. Thử vị trí cũ: `ctx_seg[start : start+len]` → nếu normalize `_`→` ` rồi so khớp thì OK (chỉ update `answer_text`)
2. Fallback: dùng regex tìm lại trong `ctx_seg` (xử lý case vị trí bị dịch do tokenizer normalize whitespace)
3. Nếu không tìm thấy → bỏ qua (sẽ bị đánh dấu impossible ở bước sau)

**Ví dụ 1 — common case (space→underscore 1:1, vị trí không đổi):**

Input:
```python
ctx_orig = ["Phạm Văn Đồng là thủ tướng"]      # length 26
ctx_seg  = ["Phạm_Văn_Đồng là thủ_tướng"]      # length 26 (space→underscore 1:1)
answers  = [{"text": ["thủ tướng"], "answer_start": [17]}]

validate_answer_in_segmented(ctx_orig, ctx_seg, answers)
```

Ouput (mutate answers):
```python
answers  = [{"text": ["thủ_tướng"], "answer_start": [17]}]
#         answer_start giữ nguyên 17 (vì prefix "Phạm_Văn_Đồng là" dài bằng "Phạm Văn Đồng là")
#         answer_text  từ "thủ tướng" → "thủ_tướng" (cập nhật compound)
```

Cơ chế bước 1:
```
ctx_seg[17:17+9] = "thủ_tướng"
"thủ_tướng".replace("_", " ") = "thủ tướng"  # trùng ans_text → OK
```

**Ví dụ 2 — vị trí thay đổi (tokenizer thêm khoảng cách giữa số và chữ):**

Input:
```python
ctx_orig = ["Đội số1 thủ tướng"]               # length 17
ctx_seg  = ["Đội số 1 thủ_tướng"]              # length 18 (thêm space sau "số")
answers  = [{"text": ["thủ tướng"], "answer_start": [8]}]

validate_answer_in_segmented(ctx_orig, ctx_seg, answers)
```

Ouput (mutate answers):
```python
answers  = [{"text": ["thủ_tướng"], "answer_start": [9]}]
#         answer_start 8 → 9  (do "số1" → "số 1" dài hơn 1 ký tự ⇒ prefix "Đội số" dài thêm 1)
#         answer_text  "thủ tướng" → "thủ_tướng"
```

---

### 1.5 `has_vietnamese(examples, language_column="language") -> bool`

Phát hiện batch có chứa tiếng Việt không.

**Cơ chế:**
1. Check cột `language` nếu có → `"vi"`
2. Fallback: regex tìm ký tự có dấu tiếng Việt (à, á, ạ, ả, ã, â, ầ, ...)

**Input:**
```python
has_vietnamese({"question": ["Hà Nội ở đâu?"], "context": ["Hà Nội là thủ đô"]})
# → True

has_vietnamese({"question": ["Where is Hanoi?"], "context": ["Hanoi is the capital"]})
# → False
```

---

## 2. `src/dataset.py`

Xử lý tokenization cho QA training/evaluation, tích hợp Vietnamese processing và quality filter.

---

### 2.1 `filter_qa_dataset(dataset, question_column, context_column, answers_column) -> Dataset`

Lọc bỏ sample chất lượng thấp từ HuggingFace Dataset. Gọi `.filter()` với `batched=False`.

**Input:**
```python
filter_qa_dataset(
    dataset=raw_dataset,          # 1000 samples
    question_column="question",
    context_column="context",
    answers_column="answers",
)
```

**Output:**
```python
# Dataset with 997 samples (3 samples bị loại vì answer_start sai / answer quá ngắn)
# In log: "Quality filter removed 3 / 1000 samples (0.3%)"
```

---

### 2.2 `prepare_train_features(examples, tokenizer, ...) -> dict[str, list]`

Tokenize batch cho training. Đây là hàm chính tích hợp tất cả xử lý tiếng Việt.

**Pipeline bên trong:**
1. `normalize_text()` từng question + context
2. `has_vietnamese()` check
3. Nếu có tiếng Việt → `segment_texts()` + `validate_answer_in_segmented()`
4. Tokenize + align start/end positions như cũ

**Input (1 sample trong batch):**
```python
{
    "question": ["Thủ đô của Việt Nam là gì?"],
    "context":  ["Hà Nội là thủ đô của Việt Nam"],
    "answers":  [{"text": ["Hà Nội"], "answer_start": [0]}],
    "is_impossible": [False],
}
```

**Output:**
```python
{
    "input_ids":       [[101, 113, 119, ..., 102]],   # token IDs
    "attention_mask":  [[1, 1, 1, ..., 1]],
    "start_positions": [5],
    "end_positions":   [7],
}
```

---

### 2.3 `prepare_eval_features(examples, tokenizer, ...) -> dict[str, list]`

Tokenize batch cho evaluation. Giống `prepare_train_features` nhưng không tính start/end positions, giữ lại `offset_mapping` và `sample_id`.

**Pipeline:**
1. `normalize_text()`
2. `has_vietnamese()` → `segment_texts()` (nếu có)
3. Tokenize
4. Giữ `offset_mapping` (set context positions = tuple, question positions = None)

---

## 3. `src/data_loader.py`

Load và build dataset từ local JSONL files.

---

### 3.1 `load_raw_datasets(config) -> DatasetDict`

Load JSONL files từ thư mục `data/` dùng HuggingFace `load_dataset`.

**Config yêu cầu:** `train_file`, `validation_file`, `test_file` (relative to `data/` hoặc absolute).

**Input config:**
```python
config.train_file = "stage2_vi_train.jsonl"
config.validation_file = "stage2_vi_validation.jsonl"
```

**Output:**
```
DatasetDict({
    train: Dataset({ features: ['id', 'question', 'context', 'answers', 'is_impossible', 'language'], num_rows: 28454 })
    validation: Dataset({ features: [...], num_rows: 3814 })
})
```

---

### 3.2 `build_qa_datasets(tokenizer, config, is_training=True) -> DatasetDict`

Pipeline chính: load → filter → tokenize → set format.

**Pipeline:**
```
raw_datasets
  │
  ├── split "train" ──→ filter_qa_dataset() ──→ .map(prepare_train_features) ──→ set_format("torch")
  │
  ├── split "validation" ──→ filter_qa_dataset() ──→ .map(prepare_train_features) ──→ set_format("torch")
  │
  └── split "test" ──→ (skip filter) ──→ .map(prepare_eval_features) ──→ set_format("torch")
```

**Output format (train/validation):**
```python
DatasetDict({
    train: Dataset({
        features: ['input_ids', 'attention_mask', 'start_positions', 'end_positions'],
        format: 'torch',
        num_rows: 28450,    # có thể giảm do quality filter
    }),
    validation: Dataset({ ... }),
})
```

**Output format (test/eval):**
```python
DatasetDict({
    test: Dataset({
        features: ['input_ids', 'attention_mask', 'offset_mapping', 'sample_id'],
        format: 'torch',
    }),
})
```

---

### 3.3 `load_dataset_for_inference(context, question, tokenizer, config) -> dict`

Chuẩn bị single sample cho inference. Gọi `prepare_eval_features` với 1 sample, trả về tensors.

**Input:**
```python
load_dataset_for_inference(
    context="Hà Nội là thủ đô của Việt Nam",
    question="Thủ đô là gì?",
    tokenizer=tokenizer,
    config=config,
)
```

**Output:**
```python
{
    "input_ids":      tensor([[101, 113, ..., 102]]),   # shape (1, seq_len)
    "attention_mask": tensor([[1, 1, ..., 1]]),
    "offset_mapping": [[(0, 0), (0, 1), ..., None, None]],
}
```
