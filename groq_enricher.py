"""
Groq API enrichment layer for Scrapbot.

Runs after intent classification. Does not replace the intent model.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

GROQ_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """You are an entity extraction engine for Scrapbot (Pakistan app).
Given the user message and detected intent, return ONLY valid JSON with this shape:
{
  "domain": "food|travel|ecommerce|jobs|automobiles|unknown",
  "entities": {
    "location": null or string,
    "budget": null or number,
    "item": null or string,
    "preference": null or string,
    "destination": null or string,
    "job_type": null or string
  },
  "language": "english|urdu|roman_urdu",
  "follow_up_question": null or string,
  "enriched_query": string,
  "confidence": 0.0 to 1.0
}
Use unknown domain and low confidence when unsure. No markdown."""


def _fallback(user_message: str, detected_intent: str) -> Dict[str, Any]:
    return {
        "domain": "unknown",
        "entities": {
            "location": None,
            "budget": None,
            "item": None,
            "preference": None,
            "destination": None,
            "job_type": None,
        },
        "language": "english",
        "follow_up_question": "Could you tell me more about what you need?",
        "enriched_query": user_message,
        "confidence": 0.0,
    }


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def enrich_message(
    user_message: str,
    detected_intent: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Enrich user message via Groq; returns fallback dict on failure."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return _fallback(user_message, detected_intent)

    history = conversation_history or []
    history_text = "\n".join(
        f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history[-6:]
    )

    user_prompt = (
        f"Detected intent: {detected_intent}\n"
        f"Recent conversation:\n{history_text or '(none)'}\n"
        f"User message: {user_message}"
    )

    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=400,
        )
        content = response.choices[0].message.content or ""
        parsed = _parse_json(content)
        if not parsed:
            return _fallback(user_message, detected_intent)
        parsed.setdefault("domain", "unknown")
        parsed.setdefault("enriched_query", user_message)
        parsed.setdefault("confidence", 0.0)
        parsed.setdefault("entities", {})
        return parsed
    except Exception:
        return _fallback(user_message, detected_intent)
