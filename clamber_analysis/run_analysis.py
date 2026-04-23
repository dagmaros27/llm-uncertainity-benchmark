import json
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bert_score import score as bertscore_score
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "clamber_benchmark.jsonl"
FIG_DIR = BASE_DIR / "figures"
OUT_DIR = BASE_DIR / "outputs"

FIG_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 140


EXPECTED_FIGURES = [
    "01_category_subclass_distribution.png",
    "02_ambiguous_split_per_category.png",
    "03_text_length_distributions.png",
    "04_cq_length_by_category.png",
    "05_confidence_distribution.png",
    "06_confusion_matrix.png",
    "07_accuracy_f1_per_category.png",
    "08_accuracy_f1_per_subclass.png",
    "09_false_negative_rate.png",
    "10_acc1_vs_acc0.png",
    "11_bertscore_distribution.png",
    "12_bertscore_by_category.png",
    "13_bertscore_by_subclass.png",
    "14_bertscore_detected_vs_missed.png",
    "15_uncertainty_type_distribution.png",
    "16_uncertainty_type_vs_subclass_heatmap.png",
    "17_silent_uncertainty_analysis.png",
    "18_cq_opening_phrases.png",
]


SUBCLASS_ORDER = [
    "whom",
    "what",
    "when",
    "where",
    "NK",
    "ICL",
    "polysemy",
    "co-reference",
]

CATEGORY_ORDER = ["MC", "FD", "LA"]
BERT_CACHE_PATH = OUT_DIR / "ambiguous_quality_cache.csv"


def print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def parse_json_maybe_nested(value):
    current = value
    for _ in range(3):
        if isinstance(current, str):
            current = current.strip()
            if not current:
                return None
            try:
                current = json.loads(current)
            except Exception:
                return current
        else:
            return current
    return current


def clean_text(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value)
    text = re.sub(r"^\s*Clarifying question:\s*", "", text, flags=re.I)
    text = re.sub(r"^\s*[-*]\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_count(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"\b\w+\b", text))


def load_dataset(path: Path) -> tuple[pd.DataFrame, int]:
    rows = []
    nested_parse_failures = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            parsed = parse_json_maybe_nested(raw_line)
            if isinstance(parsed, str):
                parsed = parse_json_maybe_nested(parsed)

            if not isinstance(parsed, dict):
                raise ValueError(f"Line {line_number} did not decode to a JSON object")

            nested = parse_json_maybe_nested(parsed.get("predict_is_ambiguous_response"))
            if not isinstance(nested, dict):
                nested_parse_failures += 1
                nested = {}

            output_value = nested.get("Output")
            confidence_value = nested.get("Confidence")
            if isinstance(output_value, str):
                lowered = output_value.strip().lower()
                if lowered in {"true", "1", "yes"}:
                    output_value = True
                elif lowered in {"false", "0", "no"}:
                    output_value = False

            try:
                confidence_value = int(confidence_value) if confidence_value is not None else np.nan
            except Exception:
                confidence_value = np.nan

            parsed["predict_output_parsed"] = output_value
            parsed["predict_confidence"] = confidence_value
            rows.append(parsed)

    df = pd.DataFrame(rows)
    for col in [
        "question",
        "context",
        "clarifying_question",
        "predict_clarifying_question",
        "category",
        "subclass",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    df["require_clarification"] = pd.to_numeric(df["require_clarification"], errors="coerce").fillna(0).astype(int)
    df["predict_ambiguous"] = pd.to_numeric(df["predict_ambiguous"], errors="coerce").fillna(0).astype(int)
    df["question_clean"] = df["question"].map(clean_text)
    df["clarifying_question_clean"] = df["clarifying_question"].map(clean_text)
    df["predict_clarifying_question_clean"] = df["predict_clarifying_question"].map(clean_text)
    df["question_wc"] = df["question_clean"].map(word_count)
    df["clarifying_question_wc"] = df["clarifying_question_clean"].map(word_count)
    df["predict_clarifying_question_wc"] = df["predict_clarifying_question_clean"].map(word_count)
    df["prediction_correct"] = (df["predict_ambiguous"] == df["require_clarification"]).astype(int)
    return df, nested_parse_failures


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, bbox_inches="tight")
    plt.close()


def classification_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def group_classification_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for name, sub in df.groupby(group_col):
        metrics = classification_metrics(sub["require_clarification"], sub["predict_ambiguous"])
        rows.append(
            {
                group_col: name,
                "n": len(sub),
                **metrics,
            }
        )
    result = pd.DataFrame(rows)
    if group_col == "category":
        result[group_col] = pd.Categorical(result[group_col], CATEGORY_ORDER, ordered=True)
        result = result.sort_values(group_col)
    elif group_col == "subclass":
        result[group_col] = pd.Categorical(result[group_col], SUBCLASS_ORDER, ordered=True)
        result = result.sort_values(group_col)
    return result.reset_index(drop=True)


def subclass_classification_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    negatives = df[df["require_clarification"] == 0].copy()
    for subclass in SUBCLASS_ORDER:
        positives = df[(df["require_clarification"] == 1) & (df["subclass"] == subclass)].copy()
        sub = pd.concat([positives, negatives], ignore_index=True)
        metrics = classification_metrics(sub["require_clarification"], sub["predict_ambiguous"])
        rows.append({"subclass": subclass, "n": len(sub), **metrics})
    return pd.DataFrame(rows)


def false_negative_rates(df: pd.DataFrame) -> pd.DataFrame:
    ambiguous = df[df["require_clarification"] == 1].copy()
    rows = []
    for subclass, sub in ambiguous.groupby("subclass"):
        fn_rate = ((sub["predict_ambiguous"] == 0).mean()) if len(sub) else np.nan
        rows.append({"subclass": subclass, "n_ambiguous": len(sub), "false_negative_rate": fn_rate})
    out = pd.DataFrame(rows).sort_values("false_negative_rate", ascending=False).reset_index(drop=True)
    return out


def get_ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def bleu_score_single(reference: str, candidate: str, max_n: int) -> float:
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    if not cand_tokens:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        cand_ngrams = Counter(get_ngrams(cand_tokens, n))
        ref_ngrams = Counter(get_ngrams(ref_tokens, n))
        total = sum(cand_ngrams.values())
        if total == 0:
            precisions.append(1e-9)
            continue
        overlap = sum(min(count, ref_ngrams[gram]) for gram, count in cand_ngrams.items())
        precisions.append((overlap + 1) / (total + 1))

    if len(cand_tokens) > len(ref_tokens):
        bp = 1.0
    else:
        bp = math.exp(1 - (len(ref_tokens) / max(len(cand_tokens), 1)))
    return bp * math.exp(sum(math.log(p) for p in precisions) / max_n)


def rouge_n_f1(reference: str, candidate: str, n: int) -> float:
    ref_ngrams = Counter(get_ngrams(reference.lower().split(), n))
    cand_ngrams = Counter(get_ngrams(candidate.lower().split(), n))
    if not ref_ngrams or not cand_ngrams:
        return 0.0
    overlap = sum(min(count, cand_ngrams[gram]) for gram, count in ref_ngrams.items())
    ref_total = sum(ref_ngrams.values())
    cand_total = sum(cand_ngrams.values())
    precision = overlap / cand_total if cand_total else 0.0
    recall = overlap / ref_total if ref_total else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def lcs_length(a_tokens, b_tokens) -> int:
    if not a_tokens or not b_tokens:
        return 0
    dp = [[0] * (len(b_tokens) + 1) for _ in range(len(a_tokens) + 1)]
    for i, a_tok in enumerate(a_tokens, start=1):
        for j, b_tok in enumerate(b_tokens, start=1):
            if a_tok == b_tok:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l_f1(reference: str, candidate: str) -> float:
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    if not ref_tokens or not cand_tokens:
        return 0.0
    lcs = lcs_length(ref_tokens, cand_tokens)
    precision = lcs / len(cand_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def classify_uncertainty_type(text: str) -> str:
    t = clean_text(text).lower()
    if not t:
        return "unclear"

    epistemic_patterns = [
        "what do you mean by",
        "could you clarify what",
        "i'm not familiar with",
        "i am not familiar with",
        "please explain",
        "what is ",
        "what does ",
        "define ",
        "mean by",
    ]
    preferential_patterns = [
        "what do you prefer",
        "which do you prefer",
        "what are your interests",
        "what are your hobbies",
        "what are your preferences",
        "any preferences",
        "budget",
        "style",
        "taste",
        "looking for",
        "would you like",
        "what kind of",
        "what type of",
        "your goals",
    ]
    aleatoric_patterns = [
        "are you referring to",
        "which one",
        "which year",
        "which version",
        "which city",
        "which location",
        "which date",
        "which country",
        "which person",
        "could you specify",
        "please specify",
        "can you specify",
        "when do you",
        "where do you",
        "what time",
        "what location",
    ]

    if any(pattern in t for pattern in epistemic_patterns):
        return "epistemic"
    if any(pattern in t for pattern in preferential_patterns):
        return "preferential"
    if any(pattern in t for pattern in aleatoric_patterns):
        return "aleatoric"

    if re.search(r"\bwhich\b", t):
        return "aleatoric"
    if re.search(r"\b(prefer|preference|interests|hobbies|budget|style)\b", t):
        return "preferential"
    if re.search(r"\b(define|meaning|explain|mean)\b", t):
        return "epistemic"
    return "unclear"


def opening_phrase(text: str, n_words: int = 4) -> str:
    tokens = re.findall(r"\b\w+\b", clean_text(text).lower())
    if not tokens:
        return ""
    return " ".join(tokens[: min(n_words, len(tokens))])


def safe_div(numerator, denominator) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    df, parse_failures = load_dataset(DATA_PATH)

    print_header("Phase 1 - EDA")
    print(f"Total entries: {len(df)}")
    missing_counts = df.replace("", np.nan).isna().sum().sort_values(ascending=False)
    print("\nMissing/null counts per field:")
    print(missing_counts.to_string())

    missing_pred_cq_when_needed = (
        (df["require_clarification"] == 1) & (df["predict_clarifying_question_clean"] == "")
    ).sum()
    print(f"\nEmpty/missing predicted CQ when clarification required: {missing_pred_cq_when_needed}")
    print(f"Failed nested parse count for predict_is_ambiguous_response: {parse_failures}")

    label_dist = df["require_clarification"].value_counts().sort_index()
    category_counts = df["category"].value_counts().reindex(CATEGORY_ORDER).fillna(0).astype(int)
    subclass_counts = df["subclass"].value_counts().reindex(SUBCLASS_ORDER).fillna(0).astype(int)
    cross_tab = pd.crosstab(df["category"], df["require_clarification"]).reindex(CATEGORY_ORDER).fillna(0).astype(int)

    print("\nrequire_clarification distribution:")
    print(label_dist.to_string())
    print("\nCategory distribution:")
    print(category_counts.to_string())
    print("\nSubclass distribution:")
    print(subclass_counts.to_string())
    print("\nCategory x require_clarification:")
    print(cross_tab.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(x=category_counts.index, y=category_counts.values, ax=axes[0], palette="deep")
    axes[0].set_title("Category Counts")
    axes[0].set_xlabel("Category")
    axes[0].set_ylabel("Count")
    sns.barplot(y=subclass_counts.index, x=subclass_counts.values, ax=axes[1], palette="deep")
    axes[1].set_title("Subclass Counts")
    axes[1].set_xlabel("Count")
    axes[1].set_ylabel("Subclass")
    savefig("01_category_subclass_distribution.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cross_pct = cross_tab.div(cross_tab.sum(axis=1), axis=0).fillna(0)
    bottom = np.zeros(len(cross_pct))
    colors = sns.color_palette("deep", n_colors=2)
    for idx, label in enumerate(sorted(cross_pct.columns)):
        axes[0].bar(cross_pct.index, cross_pct[label], bottom=bottom, label=f"label={label}", color=colors[idx])
        bottom += cross_pct[label].values
    axes[0].set_title("Ambiguous vs Not Ambiguous by Category")
    axes[0].set_ylabel("Proportion")
    axes[0].legend()
    axes[1].pie(
        [label_dist.get(0, 0), label_dist.get(1, 0)],
        labels=["Not ambiguous (0)", "Ambiguous (1)"],
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
    )
    axes[1].set_title("Overall Ambiguity Split")
    savefig("02_ambiguous_split_per_category.png")

    length_fields = {
        "question_wc": "Question",
        "clarifying_question_wc": "GT CQ",
        "predict_clarifying_question_wc": "Predicted CQ",
    }
    print("\nText length stats:")
    for field, label in length_fields.items():
        series = df[field]
        print(
            f"{label}: mean={series.mean():.2f}, median={series.median():.2f}, "
            f"min={series.min()}, max={series.max()}"
        )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, (field, label) in zip(axes, length_fields.items()):
        ax.hist(df[field], bins=30, color="#4c72b0", alpha=0.85)
        ax.set_title(f"{label} Word Count")
        ax.set_xlabel("Words")
        ax.set_ylabel("Frequency")
    savefig("03_text_length_distributions.png")

    cq_lengths = df.melt(
        id_vars=["category"],
        value_vars=["clarifying_question_wc", "predict_clarifying_question_wc"],
        var_name="cq_type",
        value_name="word_count",
    )
    cq_lengths["cq_type"] = cq_lengths["cq_type"].map(
        {
            "clarifying_question_wc": "Ground truth CQ",
            "predict_clarifying_question_wc": "Predicted CQ",
        }
    )
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=cq_lengths, x="category", y="word_count", hue="cq_type", order=CATEGORY_ORDER)
    plt.title("CQ Length by Category")
    plt.xlabel("Category")
    plt.ylabel("Word count")
    savefig("04_cq_length_by_category.png")

    confidence = df["predict_confidence"].dropna()
    print("\nConfidence stats:")
    print(confidence.describe().to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(confidence, bins=np.arange(0.5, 6.6, 1), color="#55a868", rwidth=0.85)
    axes[0].set_title("Confidence Distribution")
    axes[0].set_xlabel("Confidence")
    axes[0].set_ylabel("Count")
    for correctness, color in [(1, "#4c72b0"), (0, "#c44e52")]:
        sub = df.loc[df["prediction_correct"] == correctness, "predict_confidence"].dropna()
        axes[1].hist(
            sub,
            bins=np.arange(0.5, 6.6, 1),
            alpha=0.6,
            rwidth=0.85,
            label="Correct" if correctness == 1 else "Incorrect",
            color=color,
        )
    axes[1].set_title("Confidence by Prediction Correctness")
    axes[1].set_xlabel("Confidence")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    savefig("05_confidence_distribution.png")

    print_header("Phase 2 - Ambiguity Detection Performance")
    overall_metrics = classification_metrics(df["require_clarification"], df["predict_ambiguous"])
    print(
        "Overall metrics: "
        f"accuracy={overall_metrics['accuracy']:.4f}, "
        f"precision={overall_metrics['precision']:.4f}, "
        f"recall={overall_metrics['recall']:.4f}, "
        f"weighted_f1={overall_metrics['weighted_f1']:.4f}"
    )

    cm = confusion_matrix(df["require_clarification"], df["predict_ambiguous"], labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print(f"Confusion matrix counts: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    if fp > fn:
        skew_note = "Model skews toward over-predicting ambiguity."
    elif fn > fp:
        skew_note = "Model skews toward under-predicting ambiguity."
    else:
        skew_note = "False positives and false negatives are balanced."
    print(skew_note)

    plt.figure(figsize=(6, 5))
    cm_pct = cm / cm.sum()
    labels = np.array(
        [
            [f"TN\n{tn}\n{cm_pct[0, 0]:.1%}", f"FP\n{fp}\n{cm_pct[0, 1]:.1%}"],
            [f"FN\n{fn}\n{cm_pct[1, 0]:.1%}", f"TP\n{tp}\n{cm_pct[1, 1]:.1%}"],
        ]
    )
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1], cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    savefig("06_confusion_matrix.png")

    category_metrics = group_classification_metrics(df, "category")
    subclass_metrics = subclass_classification_metrics(df)
    print("\nPer-category metrics:")
    print(category_metrics.to_string(index=False))
    print("\nPer-subclass metrics:")
    print(subclass_metrics.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    cat_plot = category_metrics.melt(id_vars=["category"], value_vars=["accuracy", "weighted_f1"], var_name="metric", value_name="value")
    sns.barplot(data=cat_plot, x="category", y="value", hue="metric", ax=axes[0], order=CATEGORY_ORDER)
    axes[0].axhline(0.5, ls="--", color="black", linewidth=1)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Accuracy and Weighted F1 by Category")
    axes[0].set_xlabel("Category")
    axes[0].set_ylabel("Score")
    sub_plot = subclass_metrics.melt(id_vars=["subclass"], value_vars=["accuracy", "weighted_f1"], var_name="metric", value_name="value")
    sns.barplot(data=sub_plot, x="subclass", y="value", hue="metric", ax=axes[1], order=SUBCLASS_ORDER)
    axes[1].axhline(0.5, ls="--", color="black", linewidth=1)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Accuracy and Weighted F1 by Subclass")
    axes[1].set_xlabel("Subclass")
    axes[1].set_ylabel("Score")
    axes[1].tick_params(axis="x", rotation=35)
    savefig("07_accuracy_f1_per_category.png")

    plt.figure(figsize=(10, 5))
    sns.barplot(data=sub_plot, x="subclass", y="value", hue="metric", order=SUBCLASS_ORDER)
    plt.axhline(0.5, ls="--", color="black", linewidth=1)
    plt.ylim(0, 1)
    plt.title("Accuracy and Weighted F1 by Subclass")
    plt.xlabel("Subclass")
    plt.ylabel("Score")
    plt.xticks(rotation=35)
    savefig("08_accuracy_f1_per_subclass.png")

    fn_df = false_negative_rates(df)
    print("\nFalse negative rates by subclass:")
    print(fn_df.to_string(index=False))
    plt.figure(figsize=(8, 5))
    sns.barplot(data=fn_df, y="subclass", x="false_negative_rate", color="#c44e52")
    plt.title("False Negative Rate by Subclass")
    plt.xlabel("False negative rate")
    plt.ylabel("Subclass")
    savefig("09_false_negative_rate.png")

    acc1 = accuracy_score(
        df.loc[df["require_clarification"] == 1, "require_clarification"],
        df.loc[df["require_clarification"] == 1, "predict_ambiguous"],
    )
    acc0 = accuracy_score(
        df.loc[df["require_clarification"] == 0, "require_clarification"],
        df.loc[df["require_clarification"] == 0, "predict_ambiguous"],
    )
    print(f"\nAcc@1={acc1:.4f}, Acc@0={acc0:.4f}")
    plt.figure(figsize=(6, 5))
    sns.barplot(x=["Acc@1", "Acc@0"], y=[acc1, acc0], palette=["#4c72b0", "#55a868"])
    plt.ylim(0, 1)
    plt.title("Accuracy on Ambiguous vs Unambiguous Queries")
    plt.ylabel("Accuracy")
    savefig("10_acc1_vs_acc0.png")

    print_header("Phase 3 - Clarifying Question Quality")
    ambiguous_df = df[df["require_clarification"] == 1].copy()
    cache_loaded = False
    if BERT_CACHE_PATH.exists():
        cached = pd.read_csv(BERT_CACHE_PATH)
        if len(cached) == len(ambiguous_df):
            ambiguous_df = ambiguous_df.reset_index(drop=True)
            ambiguous_df[["bert_precision", "bert_recall", "bert_f1"]] = cached[
                ["bert_precision", "bert_recall", "bert_f1"]
            ]
            cache_loaded = True

    if not cache_loaded:
        ambiguous_df = ambiguous_df.reset_index(drop=True)
        ambiguous_df["bert_precision"] = 0.0
        ambiguous_df["bert_recall"] = 0.0
        ambiguous_df["bert_f1"] = 0.0

        nonempty_mask = (
            ambiguous_df["clarifying_question_clean"].str.len() > 0
        ) & (ambiguous_df["predict_clarifying_question_clean"].str.len() > 0)

        if nonempty_mask.any():
            preds = ambiguous_df.loc[nonempty_mask, "predict_clarifying_question_clean"].tolist()
            refs = ambiguous_df.loc[nonempty_mask, "clarifying_question_clean"].tolist()
            P, R, F = bertscore_score(preds, refs, lang="en", verbose=True, batch_size=16)
            ambiguous_df.loc[nonempty_mask, "bert_precision"] = P.cpu().numpy()
            ambiguous_df.loc[nonempty_mask, "bert_recall"] = R.cpu().numpy()
            ambiguous_df.loc[nonempty_mask, "bert_f1"] = F.cpu().numpy()

        ambiguous_df[["bert_precision", "bert_recall", "bert_f1"]].to_csv(BERT_CACHE_PATH, index=False)

    print(
        f"Mean BERTScore: precision={ambiguous_df['bert_precision'].mean():.4f}, "
        f"recall={ambiguous_df['bert_recall'].mean():.4f}, "
        f"f1={ambiguous_df['bert_f1'].mean():.4f}"
    )

    ambiguous_df["bleu1"] = ambiguous_df.apply(
        lambda row: bleu_score_single(row["clarifying_question_clean"], row["predict_clarifying_question_clean"], 1),
        axis=1,
    )
    ambiguous_df["bleu4"] = ambiguous_df.apply(
        lambda row: bleu_score_single(row["clarifying_question_clean"], row["predict_clarifying_question_clean"], 4),
        axis=1,
    )
    ambiguous_df["rouge1"] = ambiguous_df.apply(
        lambda row: rouge_n_f1(row["clarifying_question_clean"], row["predict_clarifying_question_clean"], 1),
        axis=1,
    )
    ambiguous_df["rougeL"] = ambiguous_df.apply(
        lambda row: rouge_l_f1(row["clarifying_question_clean"], row["predict_clarifying_question_clean"]),
        axis=1,
    )

    bleu_rouge_category = (
        ambiguous_df.groupby("category")[["bleu1", "bleu4", "rouge1", "rougeL"]].mean().reindex(CATEGORY_ORDER)
    )
    bleu_rouge_subclass = (
        ambiguous_df.groupby("subclass")[["bleu1", "bleu4", "rouge1", "rougeL"]].mean().reindex(SUBCLASS_ORDER)
    )
    print("\nBLEU/ROUGE by category:")
    print(bleu_rouge_category.to_string())
    print("\nBLEU/ROUGE by subclass:")
    print(bleu_rouge_subclass.to_string())
    bleu_rouge_category.to_csv(OUT_DIR / "bleu_rouge_by_category.csv")
    bleu_rouge_subclass.to_csv(OUT_DIR / "bleu_rouge_by_subclass.csv")

    plt.figure(figsize=(8, 5))
    plt.hist(ambiguous_df["bert_f1"], bins=30, color="#8172b3", alpha=0.9)
    plt.title("BERTScore F1 Distribution")
    plt.xlabel("BERTScore F1")
    plt.ylabel("Count")
    savefig("11_bertscore_distribution.png")

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=ambiguous_df, x="category", y="bert_f1", order=CATEGORY_ORDER)
    plt.title("BERTScore F1 by Category")
    plt.xlabel("Category")
    plt.ylabel("BERTScore F1")
    savefig("12_bertscore_by_category.png")

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=ambiguous_df, x="subclass", y="bert_f1", order=SUBCLASS_ORDER)
    plt.title("BERTScore F1 by Subclass")
    plt.xlabel("Subclass")
    plt.ylabel("BERTScore F1")
    plt.xticks(rotation=35)
    savefig("13_bertscore_by_subclass.png")

    ambiguous_df["detection_group"] = np.where(
        ambiguous_df["predict_ambiguous"] == 1, "Detected ambiguity", "Missed ambiguity"
    )
    detected_vs_missed = ambiguous_df.groupby("detection_group")["bert_f1"].agg(["mean", "median", "count"])
    print("\nBERTScore by detection group:")
    print(detected_vs_missed.to_string())
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=ambiguous_df, x="detection_group", y="bert_f1", order=["Detected ambiguity", "Missed ambiguity"])
    plt.title("BERTScore F1: Detected vs Missed Ambiguity")
    plt.xlabel("")
    plt.ylabel("BERTScore F1")
    savefig("14_bertscore_detected_vs_missed.png")

    conf_corr = ambiguous_df[["predict_confidence", "bert_f1"]].dropna()
    corr_value = conf_corr["predict_confidence"].corr(conf_corr["bert_f1"]) if len(conf_corr) > 1 else np.nan
    print(f"\nConfidence vs BERTScore correlation: {corr_value:.4f}")

    plt.figure(figsize=(7, 5))
    sns.regplot(data=conf_corr, x="predict_confidence", y="bert_f1", scatter_kws={"alpha": 0.35, "s": 25}, line_kws={"color": "red"})
    plt.title("Confidence vs BERTScore F1")
    plt.xlabel("Confidence")
    plt.ylabel("BERTScore F1")
    savefig("19_confidence_vs_bertscore.png")

    print_header("Phase 4 - Uncertainty Lens")
    df["uncertainty_type"] = df["predict_clarifying_question_clean"].map(classify_uncertainty_type)
    uncertainty_counts = df.loc[df["predict_clarifying_question_clean"] != "", "uncertainty_type"].value_counts()
    print("Uncertainty type distribution:")
    print(uncertainty_counts.to_string())

    plt.figure(figsize=(7, 5))
    sns.barplot(x=uncertainty_counts.index, y=uncertainty_counts.values, palette="deep")
    plt.title("Inferred Uncertainty Type Distribution")
    plt.xlabel("Uncertainty type")
    plt.ylabel("Count")
    savefig("15_uncertainty_type_distribution.png")

    heatmap_df = pd.crosstab(df["subclass"], df["uncertainty_type"], normalize="index").reindex(SUBCLASS_ORDER).fillna(0)
    print("\nSubclass vs uncertainty type (row-normalized):")
    print(heatmap_df.to_string())
    plt.figure(figsize=(8, 5))
    sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Subclass vs Inferred Uncertainty Type")
    plt.xlabel("Uncertainty type")
    plt.ylabel("Subclass")
    savefig("16_uncertainty_type_vs_subclass_heatmap.png")

    silent_df = df[
        (df["require_clarification"] == 1)
        & (df["predict_ambiguous"] == 0)
        & (df["predict_clarifying_question_clean"] != "")
    ].copy()
    print(f"\nSilent uncertainty cases (ambiguous, missed, but CQ present): {len(silent_df)}")
    print("Silent-case uncertainty types:")
    print(silent_df["uncertainty_type"].value_counts().to_string())
    print("\nSilent-case subclass distribution:")
    print(silent_df["subclass"].value_counts().reindex(SUBCLASS_ORDER).fillna(0).astype(int).to_string())

    detected_subclasses = ambiguous_df.loc[ambiguous_df["predict_ambiguous"] == 1, "subclass"].value_counts(normalize=True).reindex(SUBCLASS_ORDER).fillna(0)
    missed_subclasses = ambiguous_df.loc[ambiguous_df["predict_ambiguous"] == 0, "subclass"].value_counts(normalize=True).reindex(SUBCLASS_ORDER).fillna(0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    silent_uncertainty_counts = silent_df["uncertainty_type"].value_counts()
    sns.barplot(x=silent_uncertainty_counts.index, y=silent_uncertainty_counts.values, ax=axes[0], palette="deep")
    axes[0].set_title("Silent Uncertainty Types")
    axes[0].set_xlabel("Uncertainty type")
    axes[0].set_ylabel("Count")
    compare_df = pd.DataFrame(
        {
            "subclass": SUBCLASS_ORDER * 2,
            "proportion": list(missed_subclasses.values) + list(detected_subclasses.values),
            "group": ["Missed ambiguity"] * len(SUBCLASS_ORDER) + ["Detected ambiguity"] * len(SUBCLASS_ORDER),
        }
    )
    sns.barplot(data=compare_df, x="subclass", y="proportion", hue="group", ax=axes[1])
    axes[1].set_title("Subclass Distribution: Missed vs Detected")
    axes[1].set_xlabel("Subclass")
    axes[1].set_ylabel("Proportion")
    axes[1].tick_params(axis="x", rotation=35)
    savefig("17_silent_uncertainty_analysis.png")

    df["opening_phrase"] = df["predict_clarifying_question_clean"].map(opening_phrase)
    opening_counts = df.loc[df["opening_phrase"] != "", "opening_phrase"].value_counts().head(25)
    print("\nTop opening phrases:")
    print(opening_counts.to_string())
    plt.figure(figsize=(10, 8))
    sns.barplot(y=opening_counts.index, x=opening_counts.values, color="#64b5cd")
    plt.title("Top Clarifying Question Opening Phrases")
    plt.xlabel("Count")
    plt.ylabel("Opening phrase")
    savefig("18_cq_opening_phrases.png")

    sample_frames = []
    for subclass in SUBCLASS_ORDER:
        sub = df[df["subclass"] == subclass].copy()
        if len(sub):
            sample_frames.append(sub.sample(n=min(4, len(sub)), random_state=42))
    sample = pd.concat(sample_frames, ignore_index=True)[
        [
            "subclass",
            "question_clean",
            "clarifying_question_clean",
            "predict_clarifying_question_clean",
            "predict_ambiguous",
            "require_clarification",
            "uncertainty_type",
        ]
    ]
    sample.to_csv(OUT_DIR / "manual_inspection_sample.csv", index=False)
    print(f"\nSaved manual inspection sample with {len(sample)} rows to outputs/manual_inspection_sample.csv")

    print_header("Phase 5 - Model Identification Inputs")
    print(
        "Use these values against the paper tables/figures:\n"
        f"Overall accuracy={overall_metrics['accuracy']:.4f}\n"
        f"Overall weighted_f1={overall_metrics['weighted_f1']:.4f}\n"
        f"Acc@1={acc1:.4f}\n"
        f"Acc@0={acc0:.4f}"
    )
    print("\nPer-subclass metrics for Table 6 matching:")
    print(subclass_metrics[["subclass", "accuracy", "weighted_f1"]].to_string(index=False))
    confidence_nonnull = df["predict_confidence"].dropna()
    print(
        f"\nConfidence format clue: parsed JSON Output/Confidence fields on "
        f"{len(confidence_nonnull)} rows with confidence range "
        f"{int(confidence_nonnull.min()) if len(confidence_nonnull) else 'NA'}-"
        f"{int(confidence_nonnull.max()) if len(confidence_nonnull) else 'NA'}."
    )

    print_header("Phase 6 - Summary Artifacts")
    bert_by_subclass = ambiguous_df.groupby("subclass")["bert_f1"].mean().reindex(SUBCLASS_ORDER)
    dominant_uncertainty = (
        df[df["predict_clarifying_question_clean"] != ""]
        .groupby("subclass")["uncertainty_type"]
        .agg(lambda s: s.value_counts().idxmax() if len(s) else "unclear")
        .reindex(SUBCLASS_ORDER)
    )
    dataset_balance = cross_tab.copy()
    dataset_balance.columns = [f"label_{col}" for col in dataset_balance.columns]
    summary_table = (
        subclass_metrics.set_index("subclass")[["accuracy"]]
        .join(fn_df.set_index("subclass")[["false_negative_rate"]], how="left")
        .join(bert_by_subclass.rename("mean_bertscore_f1"), how="left")
        .join(dominant_uncertainty.rename("most_common_uncertainty_type"), how="left")
    ).reindex(SUBCLASS_ORDER)
    summary_table.to_csv(OUT_DIR / "phase6_key_findings_table.csv")
    dataset_balance.to_csv(OUT_DIR / "dataset_balance_per_category.csv")

    research_notes = f"""CLAMBER uncertainty notes

Does CLAMBER map onto the research uncertainty types predictably?
The mapping is partial rather than deterministic. The subclass-versus-uncertainty heatmap shows where prompt taxonomy and expressed uncertainty align, but the dominant uncertainty framing can diverge from the original CLAMBER subclass, especially when the model defaults to generic specification requests.

Which subclasses look most epistemically interesting?
The most epistemically interesting subclasses are the ones with both elevated false negative rates and a non-trivial share of epistemic-looking clarifying questions or silent uncertainty cases. These are the places where the model appears uncertain but does not reliably surface ambiguity explicitly.

What framing patterns stand out?
The opening phrase counts show whether the model tends to frame uncertainty as generic specification, preference elicitation, or concept-definition. Those opening templates should become annotation cues in the new benchmark.

What to do differently for the new dataset?
Collect or generate multiple clarification variants per prompt, preserve the model's confidence signal, and explicitly label whether the clarification is epistemic, aleatoric-discrete, or preferential-personalized. Silent-uncertainty cases deserve dedicated coverage because they reveal suppressed uncertainty rather than overt clarification behavior.
"""
    (OUT_DIR / "research_takeaways.txt").write_text(research_notes, encoding="utf-8")

    print("Saved summary table: outputs/phase6_key_findings_table.csv")
    print("Saved dataset balance table: outputs/dataset_balance_per_category.csv")
    print("Saved research notes: outputs/research_takeaways.txt")

    print("\nFigures saved:")
    for name in EXPECTED_FIGURES:
        status = "OK" if (FIG_DIR / name).exists() else "MISSING"
        print(f"- {name}: {status}")


if __name__ == "__main__":
    main()
