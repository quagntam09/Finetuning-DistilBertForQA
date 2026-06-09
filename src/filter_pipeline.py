"""
Apply full filter pipeline to all datasets and save to data/filtered_*.

Auto-generates EDA console + charts before/after filtering.

Usage:
    python src/filter_pipeline.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vietnamese import get_question_words, is_quality_sample, normalize_text

DATA_DIR = Path("data")
EDA_DIR = Path("outputs/eda")
CONFIG_PATH = Path("config/data.yaml")

_LANG_DIR_MAP = {
    "data_en": "filtered_en",
    "data_vi": "filtered_vi",
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}


def _language_and_split(path: Path) -> tuple[str | None, str]:
    rel = path.relative_to(DATA_DIR)
    parts = rel.parts
    lang_by_dir = {"data_en": "en", "data_vi": "vi"}
    lang = lang_by_dir.get(parts[0]) if parts else None
    return lang, path.stem


def _optional_int(value) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _sample_limit_from_config(path: Path, config: dict) -> tuple[int | None, str | None]:
    lang, split = _language_and_split(path)
    if lang is None:
        return None, None

    flat_key = f"{lang}_{split}_size"
    if flat_key in config:
        return _optional_int(config.get(flat_key)), flat_key

    limits = config.get("filtered_sample_limits", {})
    if isinstance(limits, dict):
        if flat_key in limits:
            return _optional_int(limits.get(flat_key)), f"filtered_sample_limits.{flat_key}"

        lang_limits = limits.get(lang, {})
        if isinstance(lang_limits, dict) and split in lang_limits:
            return _optional_int(lang_limits.get(split)), f"filtered_sample_limits.{lang}.{split}"

    return None, None


def limit_filtered_samples(rows: list[dict], path: Path, config: dict) -> tuple[list[dict], dict]:
    limit, key = _sample_limit_from_config(path, config)
    if limit is None:
        return rows, {"enabled": False, "removed": 0}
    if limit < 0:
        raise ValueError(f"{key} must be >= 0, got {limit}")

    before = len(rows)
    if before <= limit:
        return rows, {
            "enabled": True,
            "key": key,
            "limit": limit,
            "before": before,
            "after": before,
            "removed": 0,
        }

    seed = config.get("filtered_sample_seed", config.get("seed", 42))
    rng = random.Random(f"{seed}:{path.as_posix()}:{key}")
    selected_indices = sorted(rng.sample(range(before), limit))
    limited = [rows[idx] for idx in selected_indices]
    return limited, {
        "enabled": True,
        "key": key,
        "limit": limit,
        "before": before,
        "after": len(limited),
        "removed": before - len(limited),
    }


def _answer_len(row: dict) -> int:
    ans = row.get("answers", {})
    text = ans.get("text") if isinstance(ans, dict) else None
    if text and len(text) > 0 and text[0]:
        return len(str(text[0]))
    return 0


def _compute_stats(rows: list[dict], label: str) -> dict:
    ans_lens = [_answer_len(r) for r in rows]
    ctx_lens = [len(r.get("context", "")) for r in rows]
    q_lens = [len(r.get("question", "")) for r in rows]
    qword_count = 0
    lang_counts: dict[str, int] = {}
    impossible = 0
    qword_freq: dict[str, int] = {}
    no_qword = 0
    for r in rows:
        lang = r.get("language", "en")
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
        q = normalize_text(r.get("question", ""))
        words = get_question_words(q, lang)
        if words:
            qword_count += 1
            for w in words:
                qword_freq[w] = qword_freq.get(w, 0) + 1
        else:
            no_qword += 1
        if r.get("is_impossible", False):
            impossible += 1

    def _len_stats(values: list[int]) -> dict:
        if not values:
            return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0, "max": 0}
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "min": int(np.min(values)),
            "max": int(np.max(values)),
        }

    return {
        "label": label,
        "n": len(rows),
        "answer_lens": ans_lens,
        "ctx_lens": ctx_lens,
        "q_lens": q_lens,
        "lang_counts": lang_counts,
        "impossible": impossible,
        "qword_count": qword_count,
        "no_qword": no_qword,
        "qword_freq": qword_freq,
        "answer_len": _len_stats(ans_lens),
        "context_len": _len_stats(ctx_lens),
        "question_len": _len_stats(q_lens),
    }


def print_stats(before: dict, after: dict):
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  EDA — {before['label']} → {after['label']}")
    print(sep)

    print(f"  {'':25s} {'Before':>12s} {'After':>12s} {'Δ%':>8s}")
    print(f"  {'─' * 60}")
    print(f"  {'Total samples':25s} {before['n']:>12,} {after['n']:>12,} {100*(after['n']-before['n'])/before['n']:>7.1f}%")
    for lang, count in sorted(before["lang_counts"].items()):
        a = after["lang_counts"].get(lang, 0)
        print(f"  {'  └ ' + lang:25s} {count:>12,} {a:>12,} {100*(a-count)/count:>7.1f}%")
    print(f"  {'Impossible':25s} {before['impossible']:>12,} {after['impossible']:>12,} {100*(after['impossible']-before['impossible'])/before['impossible'] if before['impossible'] else 0:>7.1f}%")
    print(f"  {'Has question word':25s} {before['qword_count']:>12,} {after['qword_count']:>12,} {100*(after['qword_count']-before['qword_count'])/before['qword_count'] if before['qword_count'] else 0:>7.1f}%")
    print(f"  {'Nhóm khác (no qword)':25s} {before['no_qword']:>12,} {after['no_qword']:>12,} {100*(after['no_qword']-before['no_qword'])/before['no_qword'] if before['no_qword'] else 0:>7.1f}%")

    for name, key in [("Answer len", "answer_len"), ("Context len", "context_len")]:
        print(f"\n  {name}:")
        print(f"  {'':25s} {'Mean':>10s} {'Median':>10s} {'Std':>10s} {'Min':>8s} {'Max':>8s}")
        for d, lbl in [(before, "Before"), (after, "After")]:
            s = d[key]
            print(f"  {lbl:25s} {s['mean']:>10.1f} {s['median']:>10.1f} {s['std']:>10.1f} {s['min']:>8} {s['max']:>8}")

    # Question word frequency
    for label, d in [("Before", before), ("After", after)]:
        freq = dict(d["qword_freq"])
        no_q = d["no_qword"]
        if no_q:
            freq["Nhóm khác"] = no_q
        if freq:
            total = sum(freq.values())
            top = sorted(freq.items(), key=lambda x: -x[1])[:15]
            print(f"\n  Question words ({label}, total={total:,}):")
            for w, c in top:
                print(f"    {w:20s} {c:>8,} ({100*c/total:>5.1f}%)")


def print_sample_limit_stats(limit_stats: dict):
    if not limit_stats.get("enabled"):
        return

    print(
        f"  Filtered sample limit:    {limit_stats['before']:,} -> {limit_stats['after']:,} "
        f"({limit_stats['key']}={limit_stats['limit']:,}, removed {limit_stats['removed']:,})"
    )


def plot_stats(before: dict, after: dict, name: str):
    if after["n"] == 0:
        print("  [SKIP] no samples after filtering, skipping charts")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("whitegrid")
        plt.rcParams["figure.dpi"] = 120
    except ImportError:
        print("  [SKIP] matplotlib/seaborn not installed, skipping charts")
        return

    out_dir = EDA_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"EDA — {name} (after filter, n={after['n']:,})", fontsize=14)

    # 1. Language distribution
    lang_df = after["lang_counts"]
    colors_lang = {"en": "#3498db", "vi": "#e74c3c"}
    axes[0, 0].bar(lang_df.keys(), lang_df.values(), color=[colors_lang.get(k, "#95a5a6") for k in lang_df])
    axes[0, 0].set_title("Language Distribution")
    axes[0, 0].set_xlabel("Language")

    # 2. Impossible pie
    imp_counts = [after["n"] - after["impossible"], after["impossible"]]
    axes[0, 1].pie(imp_counts, labels=["Possible", "Impossible"], autopct="%1.1f%%", colors=["#2ecc71", "#e74c3c"])
    axes[0, 1].set_title("Impossible vs Possible")

    # 3. Context length (after)
    axes[1, 0].hist(after["ctx_lens"], bins=50, color="#9b59b6", edgecolor="white")
    axes[1, 0].set_title(f"Context Length (chars)\nmean={np.mean(after['ctx_lens']):.1f}, median={np.median(after['ctx_lens']):.0f}")
    axes[1, 0].set_xlabel("Length (chars)")

    # 4. Question word frequency (including Nhóm khác)
    qwf = dict(after["qword_freq"])
    if after["no_qword"]:
        qwf["Nhóm khác"] = after["no_qword"]
    if qwf:
        sorted_words = sorted(qwf.items(), key=lambda x: -x[1])[:15]
        words, counts = zip(*sorted_words) if sorted_words else ([], [])
        total_qw = sum(counts)
        bar_colors = ["#30c7c2" if w == "Nhóm khác" else "#e67e22" for w in words]
        bars = axes[1, 1].barh(range(len(words)), counts, color=bar_colors)
        axes[1, 1].set_yticks(range(len(words)))
        axes[1, 1].set_yticklabels(words)
        axes[1, 1].invert_yaxis()
        axes[1, 1].set_title(f"Top Question Words (after, n={total_qw:,})")
        axes[1, 1].set_xlabel("Count")
        for bar, c in zip(bars, counts):
            pct = 100 * c / total_qw
            axes[1, 1].text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                            f'{c:,} ({pct:.1f}%)', va='center', fontsize=8)
    else:
        axes[1, 1].text(0.5, 0.5, "No question words found", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_dir / "01_overview.png", bbox_inches="tight")
    plt.close()

    # Before vs After overlay
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Before vs After Filtering — {name}", fontsize=14)

    for idx, (key, title, color) in enumerate([
        ("answer_lens", "Answer Length", "#e74c3c"),
        ("ctx_lens", "Context Length", "#9b59b6"),
    ]):
        b_data = [x for x in before[key] if x > 0] if key == "answer_lens" else before[key]
        a_data = [x for x in after[key] if x > 0] if key == "answer_lens" else after[key]
        axes[idx].hist(b_data, bins=50, alpha=0.5, label=f"Before (n={len(before[key]):,})", color="gray")
        axes[idx].hist(a_data, bins=50, alpha=0.7, label=f"After (n={len(after[key]):,})", color=color)
        axes[idx].set_title(title)
        axes[idx].set_xlabel("Length (chars)")
        axes[idx].legend()

    plt.tight_layout()
    plt.savefig(out_dir / "02_before_vs_after.png", bbox_inches="tight")
    plt.close()

    print(f"  EDA charts: {out_dir}/")


def run_pipeline(path: Path):
    rel = path.relative_to(DATA_DIR)
    # Determine language subfolder
    parts = rel.parts
    lang_sub = _LANG_DIR_MAP.get(parts[0], "filtered_other")
    out_dir = Path("data") / lang_sub
    name = rel.with_suffix("").as_posix().replace("/", "_") + "_filtered"
    print(f"\n{'=' * 60}")
    print(f"  {path}  →  {out_dir / f'{name}.jsonl'}")
    print(f"{'=' * 60}")

    config = load_config()
    rows = load_jsonl(path)
    before_stats = _compute_stats(rows, "Before")

    rejected = {"quality": 0}
    kept = []

    for row in rows:
        c = normalize_text(row["context"])
        ans = row["answers"]

        if not is_quality_sample(c, ans):
            rejected["quality"] += 1
            continue
        kept.append(row)

    after_filter_count = len(kept)
    kept, limit_stats = limit_filtered_samples(kept, path, config)
    after_stats = _compute_stats(kept, "After")

    n_before = len(rows)
    n_after = len(kept)
    print(f"  Removed by quality filter: {rejected['quality']:,} ({100 * rejected['quality'] / n_before:.1f}%)")
    print_sample_limit_stats(limit_stats)
    print(f"  Total removed:             {n_before - n_after:,} ({100 * (n_before - n_after) / n_before:.1f}%)")
    if after_filter_count != n_after:
        print(f"  Final filtered samples:     {after_filter_count:,} -> {n_after:,}")

    print_stats(before_stats, after_stats)
    plot_stats(before_stats, after_stats, name)

    out_path = out_dir / f"{name}.jsonl"
    save_jsonl(kept, out_path)
    print(f"  Saved: {out_path}")


def _target_paths() -> list[Path]:
    targets: list[Path] = []
    for sub in ["data_en", "data_vi"]:
        target = DATA_DIR / sub
        if target.is_dir():
            targets.extend(sorted(target.rglob("*.jsonl")))
    if not targets:
        print(f"No JSONL files found in {DATA_DIR / 'data_en'}/ or {DATA_DIR / 'data_vi'}/")
    return targets


def main():
    jsonl_files = _target_paths()
    for path in jsonl_files:
        run_pipeline(path)

    print(f"\n{'=' * 60}")
    print(f"  Done! Saved to data/filtered_en/ and data/filtered_vi/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
