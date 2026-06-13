"""
Helper utility functions for Scrapbot.

This module contains reusable utility functions used across
different parts of the application, such as text normalization,
validation, and common data transformations.
"""

import os
import re
import difflib
from typing import Optional, List, Dict, Any


def fuzzy_match_choice(text: str, choices: List[str], cutoff: float = 0.5, n: int = 1) -> Optional[str]:
    """
    Return the best matching choice for text, or None if no good match.
    Handles typos like "software enginner" -> "software engineer", "icecreem" -> "ice cream".
    """
    if not text or not choices:
        return None
    text = text.lower().strip()
    choice_lower = [c.lower() for c in choices]
    # Exact match
    if text in choice_lower:
        return choices[choice_lower.index(text)]
    matches = difflib.get_close_matches(text, choice_lower, n=n, cutoff=cutoff)
    if not matches:
        return None
    return choices[choice_lower.index(matches[0])]


def get_project_root() -> str:
    """
    Return the project root directory (folder containing app.py, chatbot.py, data/, etc.).
    Use this so file paths work regardless of the current working directory.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def normalize_text(text: str) -> str:
    """
    Normalize user input text for consistent processing.
    
    This function:
    - Converts to lowercase for case-insensitive matching
    - Strips whitespace to remove accidental spaces
    - Normalizes multiple spaces to single space
    
    Why normalization is important:
    - User inputs vary in capitalization and spacing
    - Normalization ensures consistent matching against patterns
    - Reduces false negatives in entity extraction
    
    Args:
        text: Raw user input text
        
    Returns:
        Normalized text string
    """
    if not text:
        return ""
    
    # Convert to lowercase for case-insensitive processing
    # This ensures "Islamabad" and "islamabad" are treated the same
    normalized = text.lower()
    
    # Strip leading/trailing whitespace
    normalized = normalized.strip()
    
    # Replace multiple spaces with single space
    # Handles cases like "I  want  pizza" -> "I want pizza"
    normalized = re.sub(r'\s+', ' ', normalized)
    
    return normalized


def is_knowledge_question(text: str) -> bool:
    """
    Determine if user input is a knowledge-seeking question.
    
    Knowledge questions are queries asking for information,
    definitions, or explanations rather than transactional actions.
    These are good candidates for RAG retrieval.
    
    Why this distinction matters:
    - Knowledge questions benefit from RAG (retrieval from knowledge base)
    - Action queries (buy, order) need domain-specific handlers
    - Helps route queries to appropriate processing pipeline
    
    Args:
        text: Normalized user input
        
    Returns:
        True if text appears to be a knowledge question
    """
    knowledge_patterns = [
        "what is", "what are", "define", "explain", "tell me about",
        "information", "details", "how does", "why", "when did",
        "who is", "where is", "describe"
    ]
    
    # Check if any knowledge pattern exists in text
    # Using 'in' for substring matching (more flexible than exact match)
    return any(pattern in text for pattern in knowledge_patterns)


def is_action_query(text: str) -> bool:
    """
    Determine if user input is an action-oriented query.
    
    Action queries require domain-specific handlers rather than
    general knowledge retrieval. Examples: buying, ordering, booking.
    
    Args:
        text: Normalized user input
        
    Returns:
        True if text appears to be an action query
    """
    action_patterns = [
        "buy", "order", "purchase", "book", "reserve",
        "job", "vacancy", "hiring", "apply",
        "flight", "travel", "trip",
        "car", "vehicle", "automobile",
        "restaurant", "price", "cost"
    ]
    
    return any(pattern in text for pattern in action_patterns)


def extract_cities(text: str, city_list: List[str]) -> Optional[str]:
    """
    Extract city name from text using exact matching.
    
    This is a simple string matching approach. For more sophisticated
    entity extraction, use the spaCy-based entity extractor.
    
    Args:
        text: Text to search in
        city_list: List of valid city names
        
    Returns:
        First matching city name or None
    """
    text_lower = text.lower()
    for city in city_list:
        if city.lower() in text_lower:
            return city
    return None


def validate_confidence(confidence: float, threshold: float = 0.5) -> bool:
    """
    Validate if confidence score meets threshold.
    
    Used to determine if intent classification is reliable enough
    or if fallback to RAG is needed.
    
    Args:
        confidence: Confidence score (0.0 to 1.0)
        threshold: Minimum acceptable confidence (default 0.5)
        
    Returns:
        True if confidence meets threshold
    """
    return confidence >= threshold


def format_currency(amount: float, currency: str = "PKR") -> str:
    """
    Format monetary amounts for display.
    
    Args:
        amount: Numeric amount
        currency: Currency code (default: PKR)
        
    Returns:
        Formatted currency string (e.g., "PKR 1,50,000")
    """
    # Format with commas for readability
    # Indian/Pakistani numbering system uses lakhs (100,000)
    if amount >= 100000:
        lakhs = amount / 100000
        return f"{currency} {lakhs:.1f} lacs"
    else:
        return f"{currency} {amount:,.0f}"


def parse_price_range(price_str: str) -> float:
    """
    Parse a price range string (e.g. "Rs 350-450", "PKR 500") to a numeric value.
    Returns the first/lower number for sorting (cheapest first).
    """
    if not price_str or not isinstance(price_str, str):
        return float("inf")
    numbers = re.findall(r"[\d.]+", price_str)
    if not numbers:
        return float("inf")
    try:
        return float(numbers[0])
    except (ValueError, TypeError):
        return float("inf")


def parse_distance(distance_str: str) -> float:
    """
    Parse a distance string (e.g. "1.2 km", "2 km") to a numeric value in km.
    Used for local relevance ranking (sorting places by distance).

    Args:
        distance_str: String like "1.2 km", "2 km", "500 m"

    Returns:
        Distance in km as float, or float('inf') if unparseable (sorts last).
    """
    if not distance_str or not isinstance(distance_str, str):
        return float("inf")
    s = distance_str.strip().lower()
    try:
        if s.endswith("km"):
            return float(re.sub(r"[^\d.]", "", s) or 0)
        if s.endswith("m") and "km" not in s:
            return float(re.sub(r"[^\d.]", "", s) or 0) / 1000.0
        return float(re.sub(r"[^\d.]", "", s) or 0)
    except (ValueError, TypeError):
        return float("inf")


def safe_get(dictionary: Dict[str, Any], *keys, default=None):
    """
    Safely get nested dictionary values.
    
    Prevents KeyError when accessing nested dictionaries.
    Example: safe_get(data, 'food', 'islamabad', 'biryani', default=[])
    
    Args:
        dictionary: Dictionary to search in
        *keys: Variable number of keys to traverse
        default: Default value if key path doesn't exist
        
    Returns:
        Value at key path or default
    """
    current = dictionary
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
            if current is None:
                return default
        else:
            return default
    return current if current is not None else default
