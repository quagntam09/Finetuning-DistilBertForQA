import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from config_model import Config
from training import QATrainer


# Doi 2 dong nay neu muon chay nhanh khong can truyen tham so CLI.
DEFAULT_MODEL_DIR = "outputs/checkpoints_en/best_model"
DEFAULT_PROFILE_NAME = "train_stage2_vi_from_mixed"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Thu muc model da train, vi du: outputs/checkpoints_en/best_model",
    )
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE_NAME,
        choices=Config.available_profiles(),
        help="Profile dataset/cau hinh train moi trong config/model.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Thu muc luu model moi. Neu bo trong se lay output_dir tu profile.",
    )
    parser.add_argument(
        "--train-file",
        default=None,
        help="File train jsonl. Neu bo trong se lay train_file tu profile.",
    )
    parser.add_argument(
        "--validation-file",
        default=None,
        help="File validation jsonl. Neu bo trong se lay validation_file tu profile.",
    )
    parser.add_argument(
        "--load-optimizer-state",
        action="store_true",
        help="Load ca optimizer state. Chi nen dung khi resume cung dataset/run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    trainer = QATrainer(
        profile_name=args.profile,
        init_checkpoint_dir=args.model_dir,
        output_dir=args.output_dir,
        train_file=args.train_file,
        validation_file=args.validation_file,
        load_optimizer_state=args.load_optimizer_state or None,
    )
    trainer.train()


if __name__ == "__main__":
    main()
