# Pipeline xử lý dữ liệu QA

## Kiến trúc tổng quan

```
data/data_en/*.jsonl, data/data_vi/*.jsonl (raw)
    │
    ▼
filter_pipeline.py  ──gọi──►  vietnamese.py (is_quality_sample, has_question_word)
    │                           ▲
    ▼                           │
data/filtered_*/*_filtered.jsonl ── tự sinh EDA console/charts
    │
    ▼
data_loader.py::build_qa_datasets()
    │
    ├─ load_raw_datasets()
    │   └─ _resolve_filtered() → ưu tiên data/filtered_*/, fallback raw data/
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
| `run_pipeline(path)` | filtered JSONL + console EDA | Đọc raw → filter quality → save |
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

### training.py (train theo profile)

| Function | Output | Ghi chú |
|---|---|---|
| `QATrainer.train()` | `training_history.json`, `best_model/`, plots | Train một profile trong `config/model.yaml` |
| `run_profile_pipeline()` | `pipeline_history.json`, pipeline plots | Chạy nhiều profile liên tiếp theo `default_pipeline_profiles` hoặc `--profiles` |
| `load_checkpoint()` | model weights | Dùng `init_checkpoint_dir` để transfer giữa các profile |

### dataset.py (tokenize & align)

| Function | Input | Output |
|---|---|---|
| `prepare_train_features()` | batch + tokenizer | input_ids, attention_mask, start_positions, end_positions |
| `prepare_eval_features()` | batch + tokenizer | input_ids, attention_mask, offset_mapping, sample_id |

## Luồng sử dụng

```bash
# Bước 1: Filter + EDA (chạy 1 lần sau khi download dataset mới)
python src/filter_pipeline.py

# Bước 2: Train default pipeline
# Hiện tại: train_en -> train_vi_from_en
python src/model/training.py

# Hoặc train một profile riêng
python src/model/training.py --profile train_vi
```

## Lưu ý

- **Chỉ xử lý `data/data_en/` và `data/data_vi/`** — bỏ qua stage*, eval* trong thư mục data/.
- `data_loader.py` không còn gọi `filter_qa_dataset()` — filter tách riêng.
- `data/filtered_en/` và `data/filtered_vi/` là source chính cho training.
- Nếu profile trỏ raw path `data/data_*/*.jsonl`, data_loader sẽ ưu tiên file filtered tương ứng nếu tồn tại; nếu profile trỏ thẳng `data/filtered_*`, nó load đúng file đó.
- `is_quality_sample` lọc answer quá ngắn, không khớp context, hoặc bị cắt giữa token/từ.
- Question words hiện chỉ dùng cho thống kê EDA và question-group sampler, không dùng để loại mẫu trong `filter_pipeline.py`.
- Default training pipeline lấy từ `config/model.yaml::default_pipeline_profiles`; hiện là `train_en` rồi `train_vi_from_en`.
- Best checkpoint mặc định chọn theo `best_metric: f1`; `val_loss` vẫn được lưu để vẽ loss curve.
