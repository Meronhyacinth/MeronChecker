"""Local API for a trained MeronChecker baseline model."""
from pathlib import Path
import re
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "meronchecker_baseline.joblib"

app = FastAPI(title="MeronChecker API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class AnalyseRequest(BaseModel):
    text: str = Field(min_length=60, max_length=30_000)

def sentence_metrics(text: str):
    sentences = [s for s in re.split(r"[.!?]+", text) if len(s.split()) > 2]
    lengths = [len(s.split()) for s in sentences]
    average = round(sum(lengths) / len(lengths), 1) if lengths else 0
    variation = round((sum((x-average) ** 2 for x in lengths) / len(lengths)) ** 0.5, 1) if lengths else 0
    return {"sentences": len(sentences), "average_sentence_words": average, "sentence_length_std_dev": variation}

@app.get("/health")
def health():
    return {"ready": MODEL_PATH.exists(), "model_path": str(MODEL_PATH.name)}

@app.post("/analyse")
def analyse(request: AnalyseRequest):
    if not MODEL_PATH.exists():
        raise HTTPException(503, "Model is not trained yet. Run python training/train.py first.")
    bundle = joblib.load(MODEL_PATH)
    probability = float(bundle["model"].predict_proba(bundle["features"].transform([request.text]))[0][1])
    return {
        "ai_likelihood": round(probability * 100, 1),
        "metrics": sentence_metrics(request.text),
        "disclaimer": "AI-likelihood is an indicator, not proof of AI use or misconduct.",
        "scope": "English prose only; model performance must be checked against held-out and real-world data.",
    }
