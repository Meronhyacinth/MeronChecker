"""Train an inspectable English AI-writing baseline from public labelled datasets."""
from pathlib import Path
import json
import random

import joblib
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "artifacts"
RANDOM_SEED = 42
HC3_PER_CLASS = 6_000
MODERN_TRAIN_PER_CLASS = 4_000
MODERN_TEST_PER_CLASS = 2_000
HC3_URL = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl"
MODERN_DATASET = "silentone0725/ai-human-text-detection-v1"


def normalise(text):
    return " ".join(str(text).split()).strip()


def collect_hc3_examples():
    """Collect a balanced public sample from HC3 without its legacy loader script."""
    rows = load_dataset(
        "json",
        data_files={"train": HC3_URL},
        split="train",
        streaming=True,
    ).shuffle(seed=RANDOM_SEED, buffer_size=10_000)

    human, ai = [], []
    for row in rows:
        human.extend(
            text for answer in row.get("human_answers", [])
            if len(text := normalise(answer)) >= 120
        )
        ai.extend(
            text for answer in row.get("chatgpt_answers", [])
            if len(text := normalise(answer)) >= 120
        )
        if len(human) >= HC3_PER_CLASS and len(ai) >= HC3_PER_CLASS:
            break

    random.Random(RANDOM_SEED).shuffle(human)
    random.Random(RANDOM_SEED).shuffle(ai)
    if min(len(human), len(ai)) < 500:
        raise RuntimeError("HC3 did not provide enough usable English examples.")
    return human[:HC3_PER_CLASS], ai[:HC3_PER_CLASS]


def collect_modern_examples(split, per_class):
    """Collect a balanced slice from a separate labelled public dataset."""
    rows = load_dataset(MODERN_DATASET, split=split, streaming=True).shuffle(
        seed=RANDOM_SEED + (0 if split == "train" else 1), buffer_size=10_000
    )
    human, ai = [], []
    for row in rows:
        text = normalise(row.get("text", ""))
        label = normalise(row.get("label", "")).lower()
        if len(text) < 120:
            continue
        if label == "human" and len(human) < per_class:
            human.append(text)
        elif label == "ai" and len(ai) < per_class:
            ai.append(text)
        if len(human) >= per_class and len(ai) >= per_class:
            break

    if min(len(human), len(ai)) < 500:
        raise RuntimeError(f"{MODERN_DATASET} {split} split did not provide enough examples.")
    return human, ai


def metric_report(labels, predictions, dataset, warning):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {
        "dataset": dataset,
        "classes": {"0": "human", "1": "AI"},
        "examples": len(labels),
        "accuracy": round(accuracy_score(labels, predictions), 4),
        "ai_precision": round(precision, 4),
        "ai_recall": round(recall, 4),
        "ai_f1": round(f1, 4),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "warning": warning,
    }


def export_browser_model(features, model):
    """Export an inspectable model payload for private, in-browser scoring."""
    word_vectorizer = features.transformer_list[0][1]
    char_vectorizer = features.transformer_list[1][1]
    payload = {
        "version": "baseline-0.2.0",
        "note": "Runs locally in the browser. It is an indicator, not proof of AI use.",
        "word": {
            "vocabulary": {token: int(index) for token, index in word_vectorizer.vocabulary_.items()},
            "idf": word_vectorizer.idf_.tolist(),
        },
        "char": {
            "vocabulary": {token: int(index) for token, index in char_vectorizer.vocabulary_.items()},
            "idf": char_vectorizer.idf_.tolist(),
        },
        "coefficients": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
    }
    (MODEL_DIR / "browser_model.json").write_text(
        json.dumps(payload, separators=(",", ":")), encoding="utf-8"
    )


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    hc3_human, hc3_ai = collect_hc3_examples()
    modern_train_human, modern_train_ai = collect_modern_examples(
        "train", MODERN_TRAIN_PER_CLASS
    )
    modern_test_human, modern_test_ai = collect_modern_examples(
        "test", MODERN_TEST_PER_CLASS
    )

    texts = hc3_human + modern_train_human + hc3_ai + modern_train_ai
    labels = [0] * (len(hc3_human) + len(modern_train_human))
    labels += [1] * (len(hc3_ai) + len(modern_train_ai))

    x_train, x_internal, y_train, y_internal = train_test_split(
        texts, labels, test_size=0.20, random_state=RANDOM_SEED, stratify=labels
    )
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=45_000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=45_000, sublinear_tf=True)),
    ])
    model = LogisticRegression(max_iter=800, class_weight="balanced")
    model.fit(features.fit_transform(x_train), y_train)

    internal_predictions = model.predict(features.transform(x_internal))
    external_texts = modern_test_human + modern_test_ai
    external_labels = [0] * len(modern_test_human) + [1] * len(modern_test_ai)
    external_predictions = model.predict(features.transform(external_texts))

    report = {
        "version": "baseline-0.2.0",
        "training_sources": [
            "Hello-SimpleAI/HC3 (public streamed JSONL)",
            "silentone0725/ai-human-text-detection-v1 train split (public labelled dataset)",
        ],
        "training_examples": len(x_train),
        "internal_validation": metric_report(
            y_internal,
            internal_predictions,
            "Mixed held-out split from the two training sources",
            "This random held-out score may be optimistic because it resembles training data.",
        ),
        "external_evaluation": metric_report(
            external_labels,
            external_predictions,
            "silentone0725/ai-human-text-detection-v1 test split (not used for training)",
            "This is a stronger check than the internal split, but still does not establish universal real-world accuracy.",
        ),
        "browser_export": "artifacts/browser_model.json",
        "responsible_use": "AI-likelihood is an indicator, not proof of AI use or academic misconduct.",
    }
    joblib.dump({"features": features, "model": model}, MODEL_DIR / "meronchecker_baseline.joblib")
    export_browser_model(features, model)
    (MODEL_DIR / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
