# Finetuning DistilBERT For QA

Dự án fine-tune mô hình QA kiểu SQuAD v2 cho tiếng Việt và tiếng Anh, trọng tâm là `distilbert-base-multilingual-cased`. Pipeline hiện có các bước chính: chuẩn bị dữ liệu JSONL, lọc dữ liệu, train theo profile trong YAML, lưu checkpoint tốt nhất và sinh biểu đồ/history.

## 1. Cài đặt

Yêu cầu Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Nếu máy có CUDA, PyTorch sẽ dùng GPU khi bản `torch` đang cài hỗ trợ CUDA. Muốn ép chạy CPU, đặt `force_cpu: true` trong profile ở `config/model.yaml`.

## 2. Cấu trúc quan trọng

```text
config/data.yaml              # cấu hình lọc/sample dữ liệu
config/model.yaml             # cấu hình model, train profile, output
data/data_en/*.jsonl          # raw English QA data
data/data_vi/*.jsonl          # raw Vietnamese QA data
data/filtered_*/*.jsonl       # dữ liệu đã lọc
src/filter_pipeline.py        # lọc dữ liệu + EDA
src/model/training.py         # train DistilBERT QA
src/model/evalmodel.py        # helper metric/plot, không có CLI độc lập
scripts/generate_training_visualizations.py
experiments/phobert/          # thí nghiệm PhoBERT riêng
```

## 3. Chuẩn bị dữ liệu

Repo đang dùng JSONL local. Mỗi dòng cần có tối thiểu:

```json
{
  "id": "sample-id",
  "question": "...",
  "context": "...",
  "answers": {"text": ["..."], "answer_start": [0]},
  "is_impossible": false,
  "language": "vi"
}
```

Vị trí mặc định:

```text
data/data_en/train.jsonl
data/data_en/validation.jsonl
data/data_vi/train.jsonl
data/data_vi/validation.jsonl
data/data_vi/test.jsonl
```

Có script tải mẫu trong `src/download/`, nhưng cần lưu ý `download_squad.py` hiện ghi ra `data/data_train/`, trong khi pipeline chính mong dữ liệu tiếng Anh ở `data/data_en/`. Nếu dùng script đó, hãy chuyển hoặc chỉnh output về `data/data_en/`.

## 4. Lọc dữ liệu

Chỉnh giới hạn mẫu trong `config/data.yaml`, ví dụ:

```yaml
filtered_sample_seed: 42
en_train_size: 30000
en_validation_size: 5000
vi_validation_size: 5000
```

Chạy filter:

```bash
python src/filter_pipeline.py
```

Output:

```text
data/filtered_en/data_en_train_filtered.jsonl
data/filtered_en/data_en_validation_filtered.jsonl
data/filtered_vi/data_vi_train_filtered.jsonl
data/filtered_vi/data_vi_validation_filtered.jsonl
data/filtered_vi/data_vi_test_filtered.jsonl
outputs/eda/
```

Khi train, loader sẽ ưu tiên file filtered nếu tồn tại; nếu không có thì fallback về raw file tương ứng.

## 5. Train DistilBERT

Các profile nằm trong `config/model.yaml`. Hiện `default_pipeline_profiles` đang là:

```yaml
default_pipeline_profiles:
  - train_vi
```

Chạy default pipeline:

```bash
python src/model/training.py
```

Chạy một profile cụ thể:

```bash
python src/model/training.py --profile train_vi
python src/model/training.py --profile train_en
```

Chạy chuỗi transfer English -> Vietnamese:

```bash
python src/model/training.py --profiles train_en train_vi_from_en
```

Các profile thường dùng:

| Profile | Mục đích | Output |
|---|---|---|
| `train_vi` | Train tiếng Việt từ model gốc | `outputs/checkpoints_vi` |
| `train_en` | Train tiếng Anh từ model gốc | `outputs/checkpoints_en` |
| `train_vi_from_en` | Train tiếng Việt từ checkpoint EN | `outputs/checkpoints_vi_from_en` |
| `train_vi_answer_only` | Train VI chỉ trên mẫu có answer | `outputs/checkpoints_vi_answer_only` |
| `train_en_from_vi` | Train EN từ checkpoint VI | `outputs/checkpoints_en_from_vi` |

Mỗi run lưu:

```text
outputs/checkpoints_*/training_history.json
outputs/checkpoints_*/loss_curves.png
outputs/checkpoints_*/accuracy_f1_recall_per_epoch.png
outputs/checkpoints_*/best_model/
```

`best_model/` gồm `training_state.pt`, tokenizer, `config.yaml` và history của checkpoint tốt nhất.

## 6. Sinh biểu đồ sau train

```bash
python scripts/generate_training_visualizations.py \
  --run-dir outputs/checkpoints_vi_from_en
```

Hoặc chỉ định file history:

```bash
python scripts/generate_training_visualizations.py \
  --history outputs/checkpoints_vi/best_model/training_history.json \
  --output-dir outputs/checkpoints_vi/visualizations
```

## 7. Thí nghiệm PhoBERT

PhoBERT được tách riêng trong `experiments/phobert/`.

```bash
python experiments/phobert/train.py
```

Train một profile:

```bash
python experiments/phobert/train.py --profile train_phobert_base_vi
python experiments/phobert/train.py --profile train_phobert_base_v2_vi
```

So sánh checkpoint:

```bash
python experiments/phobert/compare.py \
  --checkpoints \
  outputs/checkpoints_vi_from_en/best_model \
  outputs/checkpoints_phobert_base_vi/best_model \
  outputs/checkpoints_phobert_base_v2_vi/best_model \
  --save-dir outputs/compare_vi_models
```

## 8. Ghi chú về output và Git

Các artifact nặng như checkpoint, `.pt`, EDA, biểu đồ và cache được ignore trong `.gitignore`. Một vài file trong `outputs/checkpoints_en/best_model/` đang được Git track sẵn như checkpoint mẫu nhỏ; không nên xóa nếu bạn vẫn muốn giữ khả năng chạy profile `train_vi_from_en` từ checkpoint EN đã có.

Tài liệu chi tiết hơn:

- `docs/current_training_pipeline.md`
- `docs/data_filtering.md`
- `docs/pipeline.md`
- `experiments/phobert/README.md`
