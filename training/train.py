"""Train an inspectable English AI-writing baseline from public HC3 data."""
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
MAX_PER_CLASS = 6_000
DATA_URL = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl"

def collect_examples():
    # HC3's repository includes a legacy loader script. Streaming the maintained
    # JSONL file avoids relying on that script and stops once our balanced sample
    # is collected, which keeps free CI usage practical.
    rows = load_dataset(
        "json",
        data_files={"train": DATA_URL},
        split="train",
        streaming=True,
    ).shuffle(seed=RANDOM_SEED, buffer_size=10_000)

    human, ai = [], []
    for row in rows:
        human.extend(
            answer.strip()
            for answer in row.get("human_answers", [])
            if len(answer.strip()) >= 120
        )
        ai.extend(
            answer.strip()
            for answer in row.get("chatgpt_answers", [])
            if len(answer.strip()) >= 120
        )
        if len(human) >= MAX_PER_CLASS and len(ai) >= MAX_PER_CLASS:
            break

    random.Random(RANDOM_SEED).shuffle(human)
    random.Random(RANDOM_SEED).shuffle(ai)
    if min(len(human), len(ai)) < 500:
        raise RuntimeError("HC3 did not provide enough usable English examples.")
    return human[:MAX_PER_CLASS], ai[:MAX_PER_CLASS]

def export_browser_model(features, model):
    """Export an inspectable model payload for private, in-browser scoring."""
    word_vectorizer = features.transformer_list[0][1]
    char_vectorizer = features.transformer_list[1][1]
    payload = {
        "version": "baseline-0.1.0",
        "note": "Runs locally in the browser. It is an indicator, not proof of AI use.",
        "word": {
            "vocabulary": word_vectorizer.vocabulary_,
            "idf": word_vectorizer.idf_.tolist(),
        },
        "char": {
            "vocabulary": char_vectorizer.vocabulary_,
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
    human, ai = collect_examples()
    texts = human + ai
    labels = [0] * len(human) + [1] * len(ai)  # 0 human, 1 AI
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=RANDOM_SEED, stratify=labels
    )
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=45_000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=45_000, sublinear_tf=True)),
    ])
    x_train_features = features.fit_transform(x_train)
    model = LogisticRegression(max_iter=800, class_weight="balanced")
    model.fit(x_train_features, y_train)
    predictions = model.predict(features.transform(x_test))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="binary", zero_division=0
    )
    report = {
        "dataset": "Hello-SimpleAI/HC3 (public, streamed JSONL)",
        "classes": {"0": "human", "1": "AI"},
        "train_examples": len(x_train),
        "test_examples": len(x_test),
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "ai_precision": round(precision, 4),
        "ai_recall": round(recall, 4),
        "ai_f1": round(f1, 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "browser_export": "artifacts/browser_model.json",
        "warning": "This score applies only to this held-out dataset. It is not a claim of real-world accuracy.",
    }
    joblib.dump({"features": features, "model": model}, MODEL_DIR / "meronchecker_baseline.joblib")
    export_browser_model(features, model)
    (MODEL_DIR / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
