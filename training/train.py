"""Train an inspectable English AI-writing baseline from 150,000 public labelled examples."""
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
TARGET_PER_CLASS = 75_000
HC3_PER_CLASS = 6_000
MODERN_TRAIN_PER_CLASS = 12_000
WIKIPEDIA_HUMAN = TARGET_PER_CLASS - HC3_PER_CLASS - MODERN_TRAIN_PER_CLASS
MDTA_AI = TARGET_PER_CLASS - HC3_PER_CLASS - MODERN_TRAIN_PER_CLASS
EXTERNAL_PER_CLASS = 2_000
TEST_ROUNDS = 20
HC3_URL = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl"
MODERN_DATASET = "silentone0725/ai-human-text-detection-v1"
MDTA_CONFIGS = ("finance", "medicine", "open_qa", "reddit_eli5", "wiki_csai")


def normalise(text):
    return " ".join(str(text).split()).strip()


def take_unique(items, limit):
    result, seen = [], set()
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
        if len(result) >= limit:
            break
    return result


def collect_hc3_examples():
    """Collect a balanced public sample from HC3 without its legacy loader script."""
    rows = load_dataset(
        "json", data_files={"train": HC3_URL}, split="train", streaming=True
    ).shuffle(seed=RANDOM_SEED, buffer_size=10_000)
    human, ai = [], []
    for row in rows:
        human.extend(text for answer in row.get("human_answers", []) if len(text := normalise(answer)) >= 120)
        ai.extend(text for answer in row.get("chatgpt_answers", []) if len(text := normalise(answer)) >= 120)
        if len(human) >= HC3_PER_CLASS and len(ai) >= HC3_PER_CLASS:
            break
    if min(len(human), len(ai)) < HC3_PER_CLASS:
        raise RuntimeError("HC3 did not provide enough usable examples.")
    return human[:HC3_PER_CLASS], ai[:HC3_PER_CLASS]


def collect_modern_examples(split, per_class):
    """Collect a balanced slice from a labelled public AI/human dataset."""
    rows = load_dataset(MODERN_DATASET, split=split, streaming=True).shuffle(
        seed=RANDOM_SEED + (0 if split == "train" else 1), buffer_size=20_000
    )
    human, ai = [], []
    for row in rows:
        text, label = normalise(row.get("text", "")), normalise(row.get("label", "")).lower()
        if len(text) < 120:
            continue
        if label == "human" and len(human) < per_class:
            human.append(text)
        elif label == "ai" and len(ai) < per_class:
            ai.append(text)
        if len(human) >= per_class and len(ai) >= per_class:
            break
    if min(len(human), len(ai)) < per_class:
        raise RuntimeError(f"{MODERN_DATASET} {split} did not provide enough examples.")
    return human[:per_class], ai[:per_class]


def collect_wikipedia_human(limit):
    """Collect human-written encyclopedia passages for broader style coverage."""
    rows = load_dataset(
        "wikimedia/wikipedia", "20231101.en", split="train", streaming=True
    ).shuffle(seed=RANDOM_SEED, buffer_size=20_000)
    samples = []
    for row in rows:
        text = normalise(row.get("text", ""))
        if len(text) >= 350:
            samples.append(text[:2_000])
        if len(samples) >= limit * 2:
            break
    samples = take_unique(samples, limit)
    if len(samples) < limit:
        raise RuntimeError("Wikipedia stream did not provide enough usable English passages.")
    return samples

def iter_model_responses(row):
    for model in (row.get("model_responses") or {}).values():
        for response in model.values():
            if isinstance(response, str):
                yield response
    # Adversarial answers are intentionally held out from training.


def collect_mdta_ai(limit):
    """Collect newer open-model outputs, keeping adversarial variants for evaluation only."""
    samples = []
    for config in MDTA_CONFIGS:
        rows = load_dataset("nsp909/MDTA", config, split="train", streaming=True).shuffle(
            seed=RANDOM_SEED, buffer_size=2_000
        )
        for row in rows:
            for response in iter_model_responses(row):
                text = normalise(response)
                if len(text) >= 120:
                    samples.append(text)
            if len(samples) >= limit * 2:
                break
        if len(samples) >= limit * 2:
            break
    random.Random(RANDOM_SEED).shuffle(samples)
    samples = take_unique(samples, limit)
    if len(samples) < limit:
        raise RuntimeError("MDTA did not provide enough usable open-model responses.")
    return samples


def metric_report(labels, predictions, dataset, warning):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    matrix = confusion_matrix(labels, predictions).tolist()
    return {
        "dataset": dataset,
        "classes": {"0": "human", "1": "AI"},
        "examples": len(labels),
        "accuracy": round(accuracy_score(labels, predictions), 4),
        "ai_precision": round(precision, 4),
        "ai_recall": round(recall, 4),
        "ai_f1": round(f1, 4),
        "human_false_positive_rate": round(matrix[0][1] / max(1, sum(matrix[0])), 4),
        "ai_false_negative_rate": round(matrix[1][0] / max(1, sum(matrix[1])), 4),
        "confusion_matrix": matrix,
        "warning": warning,
    }


def twenty_rounds(labels, predictions):
    """Estimate result stability with 20 reproducible balanced test resamples."""
    human = [index for index, label in enumerate(labels) if label == 0]
    ai = [index for index, label in enumerate(labels) if label == 1]
    results = []
    round_size = min(1_000, len(human), len(ai))
    for round_number in range(TEST_ROUNDS):
        rng = random.Random(RANDOM_SEED + round_number)
        indices = rng.sample(human, round_size) + rng.sample(ai, round_size)
        round_labels = [labels[index] for index in indices]
        round_predictions = [predictions[index] for index in indices]
        results.append(accuracy_score(round_labels, round_predictions))
    return {
        "rounds": TEST_ROUNDS,
        "examples_per_round": round_size * 2,
        "average_accuracy": round(sum(results) / len(results), 4),
        "lowest_accuracy": round(min(results), 4),
        "highest_accuracy": round(max(results), 4),
        "method": "20 reproducible balanced resamples from the held-out external test set; this measures variation, not universal real-world accuracy.",
    }


def export_browser_model(features, model):
    word_vectorizer = features.transformer_list[0][1]
    char_vectorizer = features.transformer_list[1][1]
    payload = {
        "version": "baseline-0.4.0",
        "note": "Runs locally in the browser. It is an indicator, not proof of AI use.",
        "word": {"vocabulary": {token: int(index) for token, index in word_vectorizer.vocabulary_.items()}, "idf": word_vectorizer.idf_.tolist()},
        "char": {"vocabulary": {token: int(index) for token, index in char_vectorizer.vocabulary_.items()}, "idf": char_vectorizer.idf_.tolist()},
        "coefficients": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
    }
    (MODEL_DIR / "browser_model.json").write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    hc3_human, hc3_ai = collect_hc3_examples()
    modern_human, modern_ai = collect_modern_examples("train", MODERN_TRAIN_PER_CLASS)
    wikipedia_human = collect_wikipedia_human(WIKIPEDIA_HUMAN)
    mdta_ai = collect_mdta_ai(MDTA_AI)
    external_human, external_ai = collect_modern_examples("test", EXTERNAL_PER_CLASS)

    texts = hc3_human + modern_human + wikipedia_human + hc3_ai + modern_ai + mdta_ai
    labels = [0] * TARGET_PER_CLASS + [1] * TARGET_PER_CLASS
    if len(texts) != TARGET_PER_CLASS * 2 or len(labels) != TARGET_PER_CLASS * 2:
        raise RuntimeError("Training set must contain the requested balanced example count.")

    x_train, x_internal, y_train, y_internal = train_test_split(
        texts, labels, test_size=0.20, random_state=RANDOM_SEED, stratify=labels
    )
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=35_000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=35_000, sublinear_tf=True)),
    ])
    model = LogisticRegression(max_iter=1_000, class_weight="balanced")
    model.fit(features.fit_transform(x_train), y_train)

    internal_predictions = model.predict(features.transform(x_internal))
    external_texts = external_human + external_ai
    external_labels = [0] * len(external_human) + [1] * len(external_ai)
    external_predictions = model.predict(features.transform(external_texts))

    report = {
        "version": "baseline-0.3.0",
        "algorithm": "TF-IDF word and character n-grams with balanced logistic regression",
        "training_examples": len(x_train),
        "training_set": {"human": TARGET_PER_CLASS, "ai": TARGET_PER_CLASS, "total": 100_000},
        "training_sources": [
            "Hello-SimpleAI/HC3",
            "silentone0725/ai-human-text-detection-v1 train split",
            "wikimedia/wikipedia 20231101.en passages",
            "nsp909/MDTA model_responses",
        ],
        "internal_validation": metric_report(y_internal, internal_predictions, "Mixed held-out split from training sources", "This random held-out score may be optimistic because it resembles training data."),
        "external_evaluation": metric_report(external_labels, external_predictions, "silentone0725/ai-human-text-detection-v1 test split (not used for training)", "This is stronger than an internal split, but does not establish universal real-world accuracy."),
        "twenty_test_rounds": twenty_rounds(external_labels, external_predictions),
        "browser_export": "artifacts/browser_model.json",
        "responsible_use": "AI-likelihood is an indicator, not proof of AI use or academic misconduct.",
    }
    joblib.dump({"features": features, "model": model}, MODEL_DIR / "meronchecker_baseline.joblib")
    export_browser_model(features, model)
    (MODEL_DIR / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
