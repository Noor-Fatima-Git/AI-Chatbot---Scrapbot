"""
Synonym resolution for Scrapbot (Master Plan intelligence layer).

Loads data/synonyms.json and resolves user terms to canonical values
e.g. "ISB" -> "islamabad", "phone" -> "mobile".
"""

import json
import os
from typing import Optional, Dict, Any

from utils.helpers import get_project_root
from utils.logger import logger

_synonyms_cache: Optional[Dict[str, Dict[str, str]]] = None


def load_synonyms() -> Dict[str, Dict[str, str]]:
    """Load synonyms from data/synonyms.json. Cached after first load."""
    global _synonyms_cache
    if _synonyms_cache is not None:
        return _synonyms_cache
    path = os.path.join(get_project_root(), "data", "synonyms.json")
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                _synonyms_cache = json.load(f)
            logger.logger.info("Synonyms loaded from data/synonyms.json")
        else:
            _synonyms_cache = {}
    except Exception as e:
        logger.log_error(e, "synonyms.load_synonyms")
        _synonyms_cache = {}
    return _synonyms_cache


def resolve(entity_type: str, value: Optional[str]) -> Optional[str]:
    """
    Resolve a value using the synonym map for the given entity type.

    entity_type: one of "cities", "products", "jobs", "food", "trips", "automobiles"
    value: raw extracted value (e.g. "isb", "phone")

    Returns canonical value (e.g. "islamabad", "mobile") or original value if no mapping.
    """
    if not value or not isinstance(value, str):
        return value
    v = value.strip().lower()
    if not v:
        return value
    syn = load_synonyms()
    mapping = syn.get(entity_type)
    if not mapping:
        return value
    return mapping.get(v, value)


def resolve_entities(entities: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve all known entity values in the entities dict using synonyms.
    Modifies and returns the same dict (side-effect + return).
    """
    syn = load_synonyms()
    # food_item -> food
    if "food_item" in entities and entities["food_item"]:
        key = str(entities["food_item"]).strip().lower()
        if "food" in syn and key in syn["food"]:
            entities["food_item"] = syn["food"][key]
    # food_city, city, job_city -> cities
    for k in ("food_city", "city", "job_city"):
        if k in entities and entities[k]:
            key = str(entities[k]).strip().lower()
            if "cities" in syn and key in syn["cities"]:
                entities[k] = syn["cities"][key]
    if "cities" in entities and isinstance(entities["cities"], list):
        entities["cities"] = [resolve("cities", c) or c for c in entities["cities"]]
    # job_title -> jobs
    if "job_title" in entities and entities["job_title"]:
        key = str(entities["job_title"]).strip().lower()
        if "jobs" in syn and key in syn["jobs"]:
            entities["job_title"] = syn["jobs"][key]
    # trip_place -> trips
    if "trip_place" in entities and entities["trip_place"]:
        key = str(entities["trip_place"]).strip().lower()
        if "trips" in syn and key in syn["trips"]:
            entities["trip_place"] = syn["trips"][key]
    # ecommerce_product, product -> products
    for k in ("ecommerce_product", "product"):
        if k in entities and entities[k]:
            key = str(entities[k]).strip().lower()
            if "products" in syn and key in syn["products"]:
                entities[k] = syn["products"][key]
    # auto_type -> automobiles
    if "auto_type" in entities and entities["auto_type"]:
        key = str(entities["auto_type"]).strip().lower()
        if "automobiles" in syn and key in syn["automobiles"]:
            entities["auto_type"] = syn["automobiles"][key]
    return entities
