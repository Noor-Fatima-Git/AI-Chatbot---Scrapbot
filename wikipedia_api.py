"""
Wikipedia API client – free, no API key required.

Uses the public Wikipedia REST API to fetch short summaries for
knowledge questions. Fallback when local RAG has no results.
"""

import urllib.request
import urllib.parse
import json
from typing import Optional

# Wikipedia API: https://www.mediawiki.org/wiki/API:Main_page
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"


def search_wikipedia(query: str, max_sentences: int = 4) -> Optional[str]:
    """
    Get a short plain-text summary for a query from Wikipedia.

    Uses the summary (extract) from the best-matching page.
    No API key required; respects rate limits with minimal requests.

    Args:
        query: Search/question string (e.g. "What is machine learning").
        max_sentences: Max sentences to return in the summary (default 4).

    Returns:
        Summary text or None if request failed or no result.
    """
    query = query.strip()
    if not query:
        return None

    # Remove question-style prefixes so we get a cleaner search term
    for prefix in ("what is ", "who is ", "where is ", "define ", "explain "):
        if query.lower().startswith(prefix):
            query = query[len(prefix) :].strip()
            break

    try:
        # 1) Opensearch to get a page title
        search_params = {
            "action": "opensearch",
            "search": query[:200],
            "limit": "1",
            "format": "json",
        }
        url = f"{WIKI_API_URL}?{urllib.parse.urlencode(search_params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "Scrapbot/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())

        # opensearch returns [query, [titles], [descriptions], [urls]]
        if not data or len(data) < 2 or not data[1]:
            return None
        title = data[1][0]

        # 2) Get page summary (extract)
        extract_params = {
            "action": "query",
            "prop": "extracts",
            "exintro": "1",
            "explaintext": "1",
            "exsentences": str(max_sentences),
            "titles": title,
            "format": "json",
        }
        url = f"{WIKI_API_URL}?{urllib.parse.urlencode(extract_params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "Scrapbot/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())

        pages = data.get("query", {}).get("pages", {})
        page = next(iter(pages.values()), None)
        if not page or page.get("extract") is None:
            return None

        extract = (page.get("extract") or "").strip()
        return extract if extract else None

    except Exception:
        return None
