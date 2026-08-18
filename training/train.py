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
MAX_PER_CLASS = 12_000

def collect_examples():
    dataset = load_dataset("Hello-SimpleAI/HC3", "all", split="train")
    human, ai = [], []
    for row in dataset:
        human.extend(x.strip() for x in row.get("human_answers", []) if len(x.strip()) >= 120)
        ai.extend(x.strip() for x in row.get("chatgpt_answers", []) if len(x.strip()) >= 120)
    random.Random(RANDOM_SEED).shuffle(human)
    random.Random(RANDOM_SEED).shuffle(ai)
    return human[:MAX_PER_CLASS], ai[:MAX_PER_CLASS]

def main():
    MODEL_DIR.mkdir(exist_ok=True)
    human, ai = collect_examples()
    texts = human + ai
    labels = [0] * len(human) + [1] * len(ai)  # 0 human, 1 AI
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=RANDOM_SEED, stratify=labels
    )
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=80_000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=80_000, sublinear_tf=True)),
    ])
    x_train_features = features.fit_transform(x_train)
    model = LogisticRegression(max_iter=1500, class_weight="balanced", n_jobs=None)
    model.fit(x_train_features, y_train)
    predictions = model.predict(features.transform(x_test))
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average="binary", zero_division=0)
    report = {
        "dataset": "Hello-SimpleAI/HC3 (public)",
        "classes": {"0": "human", "1": "AI"},
        "train_examples": len(x_train),
        "test_examples": len(x_test),
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "ai_precision": round(precision, 4),
        "ai_recall": round(recall, 4),
        "ai_f1": round(f1, 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "warning": "This score applies only to this held-out dataset. It is not a claim of real-world accuracy.",
    }
    joblib.dump({"features": features, "model": model}, MODEL_DIR / "meronchecker_baseline.joblib")
    (MODEL_DIR / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
