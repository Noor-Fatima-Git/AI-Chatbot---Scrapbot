"""
Train XGBoost recommendation models per domain.
Run with: py train_xgb_model.py
"""
import asyncio
import json
import os
import pickle
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from xgboost import XGBClassifier
from sqlalchemy import select, text
from db.database import engine
from db.models import DomainItem

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

DOMAINS = ["food", "jobs", "products", "automobiles", "travel"]


def extract_features(items: list[dict]) -> np.ndarray:
    """Extract numeric features from domain items metadata."""
    features = []
    for item in items:
        meta = item.get("metadata_json") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}

        # Price feature
        price = 0
        for key in ["price", "salary", "fare", "cost", "price_range"]:
            val = meta.get(key)
            if val is not None:
                try:
                    price = float(str(val).replace(",", "").replace("PKR", "").strip().split()[0])
                    break
                except Exception:
                    pass

        # Rating feature
        rating = 0
        for key in ["rating", "stars", "score"]:
            val = meta.get(key)
            if val is not None:
                try:
                    rating = float(val)
                    break
                except Exception:
                    pass

        # Title length as proxy for detail
        title_len = len(item.get("title") or "")

        # Has URL (quality signal)
        has_url = 1 if item.get("url") else 0

        # Meta richness (number of fields)
        meta_richness = len(meta)

        features.append([price, rating, title_len, has_url, meta_richness])

    return np.array(features, dtype=float)


def make_labels(features: np.ndarray) -> np.ndarray:
    """
    Create synthetic relevance labels.
    Items with higher rating + more metadata = more relevant.
    """
    scores = (
        features[:, 1] * 0.4 +       # rating
        features[:, 4] * 0.3 +       # meta richness
        features[:, 3] * 0.2 +       # has url
        (features[:, 2] / 50) * 0.1  # title length
    )
    median = np.median(scores)
    return (scores >= median).astype(int)


async def train():
    async with engine.connect() as conn:
        for domain in DOMAINS:
            result = await conn.execute(
                select(DomainItem).where(DomainItem.domain == domain)
            )
            rows = result.fetchall()

            if len(rows) < 5:
                print(f"⚠️  {domain}: not enough data ({len(rows)} items), skipping")
                continue

            items = [
                {
                    "title": r.title,
                    "url": r.url,
                    "metadata_json": r.metadata_json,
                }
                for r in rows
            ]

            X = extract_features(items)
            y = make_labels(X)

            # Replace NaN/inf
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
            )
            model.fit(X, y)

            model_path = MODELS_DIR / f"xgb_{domain}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            print(f"✅ {domain}: trained on {len(rows)} items → saved {model_path}")

    print("\n✅ All XGBoost models trained!")


if __name__ == "__main__":
    asyncio.run(train())