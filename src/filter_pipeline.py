"""
Apply full filter pipeline to all datasets and save to outputs/filtered/.

Auto-generates EDA console + charts before/after filtering.

Usage:
    python src/filter_pipeline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vietnamese import get_question_words, is_quality_sample, normalize_text

DATA_DIR = Path("data")
EDA_DIR = Path("outputs/eda")

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
        "answer_len": {"mean": float(np.mean(ans_lens)), "median": float(np.median(ans_lens)), "std": float(np.std(ans_lens)), "min": int(np.min(ans_lens)), "max": int(np.max(ans_lens))},
        "context_len": {"mean": float(np.mean(ctx_lens)), "median": float(np.median(ctx_lens)), "std": float(np.std(ctx_lens)), "min": int(np.min(ctx_lens)), "max": int(np.max(ctx_lens))},
        "question_len": {"mean": float(np.mean(q_lens)), "median": float(np.median(q_lens)), "std": float(np.std(q_lens)), "min": int(np.min(q_lens)), "max": int(np.max(q_lens))},
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


def plot_stats(before: dict, after: dict, name: str):
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
    out_dir = Path("outputs") / lang_sub
    name = rel.with_suffix("").as_posix().replace("/", "_") + "_filtered"
    print(f"\n{'=' * 60}")
    print(f"  {path}  →  {out_dir / f'{name}.jsonl'}")
    print(f"{'=' * 60}")

    rows = load_jsonl(path)
    before_stats = _compute_stats(rows, "Before")

    rejected = {"quality": 0}
    kept = []

    for row in rows:
        q = normalize_text(row["question"])
        c = normalize_text(row["context"])
        lang = row.get("language", "en")
        ans = row["answers"]

        if not is_quality_sample(c, ans):
            rejected["quality"] += 1
            continue
        kept.append(row)

    after_stats = _compute_stats(kept, "After")

    n_before = len(rows)
    n_after = len(kept)
    print(f"  Removed by quality filter: {rejected['quality']:,} ({100 * rejected['quality'] / n_before:.1f}%)")
    print(f"  Total removed:             {n_before - n_after:,} ({100 * (n_before - n_after) / n_before:.1f}%)")

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
    print(f"  Done! Saved to outputs/filtered_en/ and outputs/filtered_vi/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
