# Current Training Pipeline

Tài liệu này mô tả pipeline train hiện tại theo cấu hình trong `config/data.yaml`
và `config/model.yaml`.

## 1. Dữ liệu đầu vào

Raw data được đặt theo từng ngôn ngữ:

- `data/data_vi/train.jsonl`
- `data/data_vi/validation.jsonl`
- `data/data_vi/test.jsonl`
- `data/data_en/train.jsonl`
- `data/data_en/validation.jsonl`

Các file JSONL cần có các trường chính:

- `id`
- `question`
- `context`
- `answers.text`
- `answers.answer_start`
- `is_impossible`
- `language`

## 2. Cấu hình lọc dữ liệu

File cấu hình: `config/data.yaml`.

Các tham số đang dùng:

```yaml
filtered_sample_seed: 42
en_train_size: 30000
en_validation_size: 5000
en_test_size: 5000
vi_train_size:
vi_validation_size: 5000
vi_test_size:
```

Ý nghĩa:

- Filter chạy quality filter trước.
- Sau khi qua quality filter, mới áp dụng giới hạn số mẫu nếu có cấu hình.
- Ví dụ `en_train_size: 30000` nghĩa là lấy tối đa 30.000 mẫu train tiếng Anh sau khi lọc sạch.
- Giá trị rỗng nghĩa là không giới hạn split đó.
- Pipeline hiện không còn cân bằng theo nhóm từ khóa câu hỏi bằng `max_per_group`.

## 3. Lọc dữ liệu

Chạy:

```bash
.venv/bin/python src/filter_pipeline.py
```

Luồng xử lý:

```text
data/data_en/*.jsonl hoặc data/data_vi/*.jsonl
    -> quality filter
    -> optional sample limit từ config/data.yaml
    -> EDA before/after
    -> data/filtered_en/*_filtered.jsonl hoặc data/filtered_vi/*_filtered.jsonl
```

Output chính:

- `data/filtered_vi/data_vi_train_filtered.jsonl`
- `data/filtered_vi/data_vi_validation_filtered.jsonl`
- `data/filtered_vi/data_vi_test_filtered.jsonl`
- `data/filtered_en/data_en_train_filtered.jsonl`
- `data/filtered_en/data_en_validation_filtered.jsonl`
- `data/filtered_en/data_en_test_filtered.jsonl` nếu có raw `data/data_en/test.jsonl`

## 4. Cấu hình train

File cấu hình: `config/model.yaml`.

Default pipeline hiện tại:

```yaml
default_pipeline_profiles:
  - train_vi
```

Nghĩa là khi chạy train không truyền profile, pipeline sẽ train tiếng Việt bằng
profile `train_vi`.

Nếu muốn chạy transfer learning tiếng Anh rồi tiếng Việt, truyền profile rõ ràng:

```bash
.venv/bin/python src/model/training.py --profiles train_en train_vi_from_en
```

Các profile chính:

| Profile | Train file | Validation file | Output |
|---|---|---|---|
| `train_vi` | `data/filtered_vi/data_vi_train_filtered.jsonl` | `data/filtered_vi/data_vi_validation_filtered.jsonl` | `outputs/checkpoints_vi` |
| `train_en` | `data/filtered_en/data_en_train_filtered.jsonl` | `data/filtered_en/data_en_validation_filtered.jsonl` | `outputs/checkpoints_en` |
| `train_vi_from_en` | `data/filtered_vi/data_vi_train_filtered.jsonl` | `data/filtered_vi/data_vi_validation_filtered.jsonl` | `outputs/checkpoints_vi_from_en` |
| `train_en_from_vi_eval_vi` | `data/filtered_en/data_en_train_filtered.jsonl` | `data/filtered_vi/data_vi_validation_filtered.jsonl` | `outputs/checkpoints_en_from_vi_eval_vi` |

Các profile phụ:

| Profile | Mục đích |
|---|---|
| `train_vi_answer_only` | Train tiếng Việt chỉ trên mẫu có answer, tắt tune no-answer threshold |
| `train_vi_from_en_answer_only` | Khởi tạo từ checkpoint EN rồi train VI answer-only |
| `train_en_from_vi` | Khởi tạo từ checkpoint VI rồi train EN, eval trên validation EN |
| `eval_vi`, `eval_en`, `eval_vi_test` | Profile cấu hình split eval; hiện chưa có CLI eval độc lập trong `evalmodel.py` |

## 5. Chạy train

Chạy toàn bộ default pipeline:

```bash
.venv/bin/python src/model/training.py
```

Chạy một profile:

```bash
.venv/bin/python src/model/training.py --profile train_vi
```

Chạy một chuỗi profile cụ thể:

```bash
.venv/bin/python src/model/training.py --profiles train_en train_vi_from_en
```

## 6. Logic train/eval trong mỗi epoch

Mỗi epoch thực hiện:

```text
train one epoch
    -> tính train_loss
evaluate validation loss
    -> tính val_loss
evaluate QA metrics nếu track_eval_metrics=true
    -> exact_match / accuracy / precision / recall / F1
    -> has_answer_f1 / no_answer_exact
    -> tune no-answer threshold trên validation nếu tune_no_answer_threshold=true
save best checkpoint
    -> chọn theo best_metric=f1
    -> lưu threshold tốt nhất vào best_model/config.yaml
early stopping
    -> theo metric đang chọn best và early_stopping_min_delta
```

Cấu hình quan trọng:

```yaml
track_eval_metrics: true
tune_no_answer_threshold: true
no_answer_threshold: 0.0
save_best_model: true
best_metric: f1
load_best_model: true
```

Hiện best checkpoint được chọn theo F1, không còn chọn mặc định theo `val_loss`.
`val_loss` vẫn được lưu để theo dõi loss curve.

## 7. Output sau train

Mỗi profile tạo output trong thư mục `output_dir` tương ứng:

```text
outputs/checkpoints_en/
    training_history.json
    loss_curves.png
    accuracy_f1_recall_per_epoch.png
    best_model/
        config.yaml
        training_state.pt
        tokenizer files
        training_history.json
        loss_curves.png
        accuracy_f1_recall_per_epoch.png

outputs/checkpoints_vi_from_en/
    pipeline_history.json
    pipeline_loss_curves.png
    pipeline_accuracy_f1_recall.png
    training_history.json
    best_model/
```

`training_history.json` có thể gồm:

- `train_loss`
- `val_loss`
- `em`
- `accuracy`
- `precision`
- `recall`
- `f1`
- `has_answer_f1`
- `no_answer_exact`
- `no_answer_threshold`

## 8. Eval checkpoint

`src/model/evalmodel.py` hiện là module helper để tính metric và vẽ biểu đồ, chưa
có CLI độc lập. Vì vậy các lệnh dạng sau không còn đúng với code hiện tại:

```bash
.venv/bin/python src/model/evalmodel.py --checkpoint ...
```

Hiện có thể so sánh/evaluate nhiều checkpoint bằng script trong thí nghiệm
PhoBERT:

```bash
.venv/bin/python experiments/phobert/compare.py \
  --checkpoints \
  outputs/checkpoints_vi_from_en/best_model \
  outputs/checkpoints_phobert_base_vi/best_model \
  outputs/checkpoints_phobert_base_v2_vi/best_model \
  --save-dir outputs/compare_vi_models
```

Script này đọc `config.yaml` trong từng `best_model`, dùng
`no_answer_threshold` đã lưu trong checkpoint và ghi kết quả vào `save_dir`.

## 9. Biểu đồ đánh giá

Training tự tạo các biểu đồ chuẩn trong `output_dir` và `best_model`:

- `loss_curves.png`
- `accuracy_f1_recall_per_epoch.png`

Có thể tạo thêm dashboard từ history:

```bash
.venv/bin/python scripts/generate_training_visualizations.py \
  --run-dir outputs/checkpoints_vi_from_en
```

Output mặc định nằm trong `outputs/checkpoints_vi_from_en/visualizations/`.

## 10. Cách đọc kết quả hiện tại

Nếu F1/EM xấp xỉ tỷ lệ no-answer của validation, mô hình có thể đang học cách
trả lời rỗng/no-answer quá nhiều. Khi đó cần xem thêm `training_history.json`,
`has_answer_f1`, `no_answer_exact`, `no_answer_threshold` và biểu đồ
`accuracy_f1_recall_per_epoch.png` trước khi kết luận mô hình tốt.
