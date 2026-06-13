"""
Optional free LLM client – Groq (free tier).

Uses Groq's free API with a fast model to make replies more natural.
Set GROQ_API_KEY in environment (or .env) to enable; if unset, all
methods no-op and return None so the app works without any key.
"""

import os
from typing import Optional

# Optional: load from .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"  # Free tier, fast
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def is_available() -> bool:
    """Return True if a Groq API key is set."""
    return bool(GROQ_API_KEY and GROQ_API_KEY.strip())


def complete(
    user_message: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 180,
) -> Optional[str]:
    """
    Get a completion from Groq (free tier). No-op if GROQ_API_KEY not set.

    Args:
        user_message: The user's message or question.
        system_prompt: Optional system instruction (e.g. "You are a helpful assistant.").
        max_tokens: Max response length.

    Returns:
        Reply text or None if disabled, error, or no content.
    """
    if not is_available():
        return None

    try:
        import urllib.request
        import json

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        body = json.dumps(
            {"model": GROQ_MODEL, "messages": messages, "max_tokens": max_tokens}
        ).encode()
        req = urllib.request.Request(
            GROQ_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())

        choice = (data.get("choices") or [None])[0]
        if not choice:
            return None
        content = (choice.get("message") or {}).get("content")
        return (content or "").strip() or None

    except Exception:
        return None


def enhance_reply(domain_reply: str, user_input: str) -> Optional[str]:
    """
    Use the LLM to make a short domain reply sound more natural.
    Returns None if LLM is disabled or fails (caller should use domain_reply).
    """
    if not domain_reply or not is_available():
        return None

    system_prompt = (
        "You are a friendly chatbot assistant. Rewrite the following bot reply "
        "to be a bit warmer and more natural. Keep it concise (1-4 sentences). "
        "Do not add new facts or change the meaning. Reply only with the rewritten text."
    )
    user_message = f"User asked: {user_input}\n\nCurrent bot reply:\n{domain_reply}"
    return complete(user_message, system_prompt=system_prompt, max_tokens=120)


def generate_reply_from_context(
    user_input: str,
    context_type: str,
    context_text: str,
) -> Optional[str]:
    """
    Generate a natural reply from domain or RAG context using the LLM.
    Use this as the primary reply when LLM is available (stronger than enhance_reply).

    Args:
        user_input: What the user asked.
        context_type: "domain_reply" or "rag".
        context_text: The factual reply text or RAG-retrieved chunks.

    Returns:
        LLM-generated reply or None if disabled/failed.
    """
    if not context_text or not context_text.strip() or not is_available():
        return None

    if context_type == "rag":
        system_prompt = (
            "You are a helpful assistant. Using ONLY the following retrieved context, "
            "answer the user's question in 1-4 clear sentences. Do not invent facts; "
            "if the context does not contain the answer, say so briefly. Reply only with the answer."
        )
        user_message = f"Context:\n{context_text[:2500]}\n\nUser question: {user_input}"
    else:
        # domain_reply
        system_prompt = (
            "You are a friendly chatbot. Given the following factual bot reply, "
            "respond to the user in a natural, warm way in 1-4 sentences. "
            "Do not add new facts or change the meaning. Reply only with your response."
        )
        user_message = f"User asked: {user_input}\n\nFactual bot reply:\n{context_text}"

    return complete(user_message, system_prompt=system_prompt, max_tokens=200)
