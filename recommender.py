"""
Upgraded Domain-aware recommendation system using PostgreSQL + XGBoost.
Now with query-relevant filtering and clickable links.
"""
import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from context.context_manager import context_manager
from utils.logger import logger

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _load_model(domain: str):
    path = MODELS_DIR / f"xgb_{domain}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def _clean_price(val) -> Optional[float]:
    if val is None:
        return None
    try:
        s = str(val).replace("PKR", "").replace("Rs", "").replace(",", "").replace(" ", "")
        nums = re.findall(r"[\d.]+", s)
        if nums:
            return float(nums[0])
    except Exception:
        pass
    return None


def _extract_features(items: list) -> np.ndarray:
    features = []
    for item in items:
        meta = item.get("metadata_json") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}

        price = 0
        for key in ["price", "salary", "fare", "cost"]:
            val = _clean_price(meta.get(key))
            if val is not None:
                price = val
                break

        rating = 0
        for key in ["rating", "stars", "score"]:
            val = meta.get(key)
            if val is not None:
                try:
                    rating = float(val)
                    break
                except Exception:
                    pass

        title_len = len(item.get("title") or "")
        has_url = 1 if item.get("url") else 0
        meta_richness = len(meta)
        features.append([price, rating, title_len, has_url, meta_richness])

    return np.nan_to_num(np.array(features, dtype=float), nan=0.0)


def _format_item(domain: str, item: dict, show_url: bool = True) -> str:
    meta = item.get("metadata_json") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}

    title = item.get("title") or "Unknown"
    url = item.get("url") or meta.get("url") or ""

    def clean(s):
        return str(s).replace("\u00e2\u0080\u0093", "-").replace("\u2013", "-").replace("\u2014", "-")

    if domain == "food":
        area = meta.get("area") or meta.get("location") or ""
        price = clean(meta.get("price_range") or meta.get("price") or "")
        rating = meta.get("rating") or ""
        rating_txt = f" ⭐{rating}" if rating else ""
        line = f"🍕 {title} - {area} · {price}{rating_txt}"
        return line

    if domain == "jobs":
        city = str(meta.get("city") or "").title()
        salary = meta.get("salary")
        salary_txt = f" · PKR {salary:,}" if isinstance(salary, int) else f" · {salary}" if salary else ""
        line = f"💼 {title} in {city}{salary_txt}"
        if show_url and url:
            line += f"\n   🔗 {url}"
        return line

    if domain == "products":
        price = clean(meta.get("price") or "")
        retailer = meta.get("retailer") or ""
        line = f"🛍️ {title} - {price} @ {retailer}"
        if show_url and url:
            line += f"\n   🔗 {url}"
        return line

    if domain == "automobiles":
        price = clean(meta.get("price") or "")
        line = f"🚗 {title} - {price}"
        if show_url and url:
            line += f"\n   🔗 {url}"
        return line

    if domain == "travel":
        about = str(meta.get("about") or meta.get("description") or "")[:60]
        return f"✈️ {title} - {about}"

    if domain == "real_estate":
        area = meta.get("area") or ""
        price = clean(meta.get("price") or "")
        return f"🏠 {title} - PKR {price} ({area})"

    return f"• {title}"


def _get_items_sync(domain: str, query_filter: str = "", limit: int = 100) -> list:
    """Fetch items from PostgreSQL with optional keyword filter."""
    try:
        import psycopg2
        import psycopg2.extras
        import os
        url = os.getenv("POSTGRES_URL_SYNC")
        conn = psycopg2.connect(url)
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        if query_filter:
            # Filter by title keyword for relevance
            cur.execute(
                """SELECT title, url, metadata_json FROM domain_items
                   WHERE domain=%s AND LOWER(title) LIKE %s
                   LIMIT %s""",
                (domain, f"%{query_filter.lower()}%", limit)
            )
            rows = cur.fetchall()
            # If no keyword match, fall back to all items
            if not rows:
                cur.execute(
                    "SELECT title, url, metadata_json FROM domain_items WHERE domain=%s LIMIT %s",
                    (domain, limit)
                )
                rows = cur.fetchall()
        else:
            cur.execute(
                "SELECT title, url, metadata_json FROM domain_items WHERE domain=%s LIMIT %s",
                (domain, limit)
            )
            rows = cur.fetchall()

        cur.close()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.log_error(e, "recommender._get_items_sync")
        return []


def _get_query_filter(domain: str, merged: dict) -> str:
    """Get relevant keyword to filter DB items by domain."""
    if domain == "jobs":
        return merged.get("job_title") or ""
    if domain == "products":
        return merged.get("ecommerce_product") or merged.get("product_category") or ""
    if domain == "automobiles":
        return merged.get("auto_type") or merged.get("auto_brand") or ""
    if domain == "food":
        return merged.get("food_item") or ""
    if domain == "real_estate":
        return merged.get("property_type") or ""
    if domain == "travel":
        return merged.get("trip_place") or ""
    return ""


def get_recommendations(
    domain: Optional[str],
    entities: Optional[Dict[str, Any]] = None,
    limit: int = 5,
    prefer_cheap: bool = False,
) -> List[str]:
    if not domain:
        return []

    entities = entities or {}
    ctx = context_manager.get_all_entities()
    merged = {**{k: v for k, v in ctx.items() if v}, **entities}

    try:
        # Get relevant query filter
        query_filter = _get_query_filter(domain, merged)
        items = _get_items_sync(domain, query_filter=query_filter, limit=100)
        if not items:
            return []

        model = _load_model(domain)
        if model is not None:
            X = _extract_features(items)
            try:
                probs = model.predict_proba(X)[:, 1]
            except Exception:
                probs = np.ones(len(items))
        else:
            probs = np.ones(len(items))

        # City boost
        city = (
            merged.get("food_city") or
            merged.get("job_city") or
            merged.get("city") or
            merged.get("user_location") or ""
        )
        if city:
            city = city.lower()
            boosted = []
            for i, item in enumerate(items):
                meta = item.get("metadata_json") or {}
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except Exception:
                        meta = {}
                item_city = str(meta.get("city") or meta.get("area") or meta.get("location") or "").lower()
                boost = 0.3 if city in item_city or item_city in city else 0.0
                boosted.append(probs[i] + boost)
            probs = np.array(boosted)

        # Budget filter
        budget = merged.get("budget")
        if budget is not None:
            try:
                budget = float(budget)
                filtered_items, filtered_probs = [], []
                for i, item in enumerate(items):
                    meta = item.get("metadata_json") or {}
                    if isinstance(meta, str):
                        try:
                            meta = json.loads(meta)
                        except Exception:
                            meta = {}
                    price = None
                    if domain == "automobiles":
                        val = meta.get("price")
                        if val:
                            nums = re.findall(r"[\d.]+", str(val).replace(",", ""))
                            if nums:
                                price = float(nums[0]) * 100000
                    elif domain in ("products", "food"):
                        price = _clean_price(meta.get("price") or meta.get("price_range"))
                    elif domain == "jobs":
                        val = meta.get("salary")
                        if val:
                            try:
                                price = float(val) if isinstance(val, (int, float)) else float(str(val).replace(",", ""))
                            except Exception:
                                pass
                    elif domain == "real_estate":
                        val = meta.get("price")
                        if val:
                            s = str(val).replace(",", "").lower()
                            nums = re.findall(r"[\d.]+", s)
                            if nums:
                                price = float(nums[0])
                                if "crore" in s:
                                    price *= 10000000
                                elif "lac" in s:
                                    price *= 100000

                    if price is None or price <= budget:
                        filtered_items.append(item)
                        filtered_probs.append(probs[i])

                if filtered_items:
                    items = filtered_items
                    probs = np.array(filtered_probs)
            except Exception:
                pass

        # Top N
        top_indices = np.argsort(probs)[::-1][:limit]
        results = []
        for i in top_indices:
            formatted = _format_item(domain, items[i], show_url=True)
            if formatted:
                results.append(formatted)

        return results

    except Exception as e:
        logger.log_error(e, "recommender.get_recommendations")
        return []


def append_recommendations_to_reply(
    reply: str,
    domain: Optional[str],
    entities: Optional[Dict[str, Any]] = None,
    limit: int = 5,
) -> str:
    recs = get_recommendations(domain, entities=entities, limit=limit)
    if not recs:
        return reply
    block = "\n".join(recs)
    if block in reply:
        return reply
    return f"{reply}\n\nTop recommendations:\n{block}"
