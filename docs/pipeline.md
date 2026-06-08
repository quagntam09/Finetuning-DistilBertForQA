# Pipeline xử lý dữ liệu QA

## Kiến trúc tổng quan

```
data/*.jsonl (raw)
    │
    ▼
filter_pipeline.py  ──gọi──►  vietnamese.py (is_quality_sample, has_question_word)
    │                           ▲
    ▼                           │
output/filtered/*_filtered.jsonl ── tự sinh EDA console
    │
    ▼
data_loader.py::build_qa_datasets()
    │
    ├─ load_raw_datasets()
    │   └─ _resolve_filtered() → ưu tiên output/filtered/, fallback data/
    │
    └─ dataset.map(prepare_train/eval_features)
         └─ vietnamese.py (normalize_text, segment_texts)
              └─ HuggingFace tokenizer → input_ids, start/end positions
```

## File → Function → Output

### vietnamese.py (utility core)

| Function | Input | Output | Ghi chú |
|---|---|---|---|
| `normalize_text()` | `str` | `str` | NFKC + xoá noise (HTML, URL, zero-width) |
| `is_quality_sample()` | context, answer dict | `bool` | answer length 2-9999, khớp context |
| `has_question_word()` | question, language | `bool` | EN: wh-words, VI: từ để hỏi |
| `segment_texts()` | `list[str]` | `list[str]` | Word segmentation bằng underthesea (VI) |
| `has_vietnamese()` | batch dict | `bool` | Nhận diện tiếng Việt qua cột language hoặc ký tự |

### filter_pipeline.py (chạy 1 lần, sinh filtered data)

| Function | Output | Ghi chú |
|---|---|---|
| `run_pipeline(path)` | filtered JSONL + console EDA | Đọc raw → filter quality + qword → save |
| `_compute_stats(rows)` | dict | answer_len, context_len, question_len, impossible, qword coverage |
| `print_stats(before, after)` | console table | So sánh Before/After cho từng dataset |

Chạy: `python src/filter_pipeline.py`

### data_loader.py (load & tokenize)

| Function | Output | Ghi chú |
|---|---|---|
| `_resolve_filtered(path)` | `str \| None` | Map raw path → filtered path nếu tồn tại |
| `load_raw_datasets(config)` | `DatasetDict` | Tự động chọn filtered > raw |
| `build_qa_datasets(tokenizer, config)` | `DatasetDict` (tokenized) | Gọi prepare_train/eval_features |
| `load_dataset_for_inference()` | dict tensors | Single sample inference |

### dataset.py (tokenize & align)

| Function | Input | Output |
|---|---|---|
| `prepare_train_features()` | batch + tokenizer | input_ids, attention_mask, start_positions, end_positions |
| `prepare_eval_features()` | batch + tokenizer | input_ids, attention_mask, offset_mapping, sample_id |

## Luồng sử dụng

```bash
# Bước 1: Filter + EDA (chạy 1 lần sau khi download dataset mới)
python src/filter_pipeline.py

# Bước 2: Train (data_loader tự động dùng file filtered)
python src/model/train.py
```

## Lưu ý

- **Chỉ xử lý `data/data_en/` và `data/data_vi/`** — bỏ qua stage*, eval* trong thư mục data/.
- `data_loader.py` không còn gọi `filter_qa_dataset()` — filter tách riêng.
- `output/filtered/` là single source of truth cho training.
- Nếu chưa chạy filter, data_loader fallback về raw (có warning).
- `is_quality_sample` chỉ lọc answer <2 ký tự hoặc không khớp context (bỏ giới hạn độ dài tối đa).
- `config/model.yaml` giữ nguyên profiles cũ (stage*, eval*).
