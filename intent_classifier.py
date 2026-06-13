"""
Upgraded Intent Classifier for Scrapbot.
Better keyword matching, more intents, higher confidence.
"""

import pickle
import os
from typing import Tuple, Optional, List
import numpy as np
from utils.logger import logger

# Expanded keyword intents with more coverage
_KEYWORD_INTENTS: List[Tuple[str, List[str]]] = [
    ("food", [
        "hungry", "food", "eat", "restaurant", "biryani", "burger",
        "pizza", "dine", "khana", "khaana", "order food", "delivery",
        "karahi", "nihari", "chinese", "bbq", "sundae", "shake",
        "sasti", "sasta", "ice cream", "cafe", "eatery"
    ]),
    ("job", [
        "job", "jobs", "career", "hire", "salary", "resume", "engineer",
        "work", "employment", "vacancy", "position", "internship",
        "nokri", "naukri", "kaam", "developer", "manager", "analyst",
        "remote", "part time", "full time", "hiring", "apply"
    ]),
    ("travel", [
        "flight", "flights", "trip", "trips", "hotel", "vacation",
        "travel", "visit", "tour", "booking", "hunza", "swat",
        "murree", "naran", "skardu", "tourism", "destination",
        "ticket", "airline", "passport", "visa"
    ]),
    ("ecommerce", [
        "shop", "purchase", "laptop", "mobile", "product",
        "headphones", "phone", "tablet", "camera", "watch", "tv",
        "speaker", "price", "online shopping", "daraz", "amazon",
        "order", "delivery", "khareedna", "electronic", "gadget"
    ]),
    ("automobiles", [
        "car", "bike", "vehicle", "automobile", "sedan", "suv",
        "hatchback", "electric car", "honda", "toyota", "suzuki",
        "gaari", "gari", "gaadi", "petrol", "diesel", "hybrid",
        "pakwheels", "rent a car", "driving", "auto"
    ]),
    ("real_estate", [
        "rent", "flat", "house", "property", "real estate", "apartment",
        "plot", "home", "villa", "studio", "bedroom", "zameen",
        "ghar", "makaan", "makan", "buy house", "lease", "rental",
        "buy property", "buy flat", "buy home", "buy apartment",
        "property in", "ghar chahiye", "makaan chahiye",
        "home in", "house in", "flat in",
    ]),
]


def _sklearn_available() -> bool:
    try:
        import sklearn
        return True
    except Exception:
        return False


def _keyword_classify(text: str) -> Tuple[Optional[str], float]:
    """Enhanced keyword intent classifier."""
    t = text.lower()
    scores = {}
    for intent, keywords in _KEYWORD_INTENTS:
        hits = sum(1 for kw in keywords if kw in t)
        if hits > 0:
            scores[intent] = hits

    if not scores:
        return None, 0.0

    best_intent = max(scores, key=scores.get)
    best_score = scores[best_intent]

    # Higher confidence with more keyword hits
    confidence = min(0.60 + 0.08 * best_score, 0.90)
    return best_intent, confidence


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)


class IntentClassifier:

    def __init__(self, model_path: str = None,
                 vectorizer_path: str = None,
                 confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.vectorizer = None
        self._use_transformer = False
        self._transformer_model = None

        # Try transformer model
        transformer_path = os.path.join(_PROJECT_ROOT, "intent_model_transformer.pkl")
        try:
            if os.path.exists(transformer_path):
                with open(transformer_path, "rb") as f:
                    self._transformer_model = pickle.load(f)
                self._use_transformer = True
                logger.logger.info(f"Transformer intent classifier loaded")
        except Exception as e:
            logger.log_error(e, "IntentClassifier.__init__ (transformer)")

        if model_path is None:
            model_path = os.path.join(_PROJECT_ROOT, "intent_model.pkl")
        if vectorizer_path is None:
            vectorizer_path = os.path.join(_PROJECT_ROOT, "vectorizer.pkl")

        # Try TF-IDF + LR model
        if not self._use_transformer:
            if not _sklearn_available():
                logger.logger.warning("scikit-learn not available. Using keyword intent fallback.")
            else:
                try:
                    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                        with open(vectorizer_path, "rb") as f:
                            self.vectorizer = pickle.load(f)
                        with open(model_path, "rb") as f:
                            self.model = pickle.load(f)
                        logger.logger.info(f"Intent classifier loaded from {model_path}")
                except Exception as e:
                    logger.log_error(e, "IntentClassifier.__init__")

    def classify(self, text: str) -> Tuple[Optional[str], float]:
        """Classify intent with transformer → TF-IDF → keyword fallback."""

        # Transformer path
        if self._use_transformer and self._transformer_model is not None:
            try:
                from nlp.embeddings import embedding_generator
                if embedding_generator.model is not None:
                    emb = embedding_generator.generate_embeddings([text])
                    if emb is not None and len(emb) > 0:
                        emb = np.asarray(emb, dtype=np.float64).reshape(1, -1)
                        probs = self._transformer_model.predict_proba(emb)[0]
                        intent_idx = int(np.argmax(probs))
                        intent = self._transformer_model.classes_[intent_idx]
                        confidence = float(probs[intent_idx])

                        # Boost with keyword check
                        kw_intent, kw_conf = _keyword_classify(text)
                        if kw_intent == intent:
                            confidence = min(confidence + 0.1, 0.95)

                        logger.log_intent(intent, confidence)
                        return intent, confidence
            except Exception as e:
                logger.log_error(e, "IntentClassifier.classify (transformer)")

        # TF-IDF + LR path
        if self.model is not None and self.vectorizer is not None:
            try:
                X = self.vectorizer.transform([text])
                probs = self.model.predict_proba(X)[0]
                intent_idx = np.argmax(probs)
                intent = self.model.classes_[intent_idx]
                confidence = float(probs[intent_idx])

                # Boost with keyword check
                kw_intent, kw_conf = _keyword_classify(text)
                if kw_intent == intent:
                    confidence = min(confidence + 0.1, 0.95)
                elif kw_intent and kw_conf > confidence:
                    intent = kw_intent
                    confidence = kw_conf

                logger.log_intent(intent, confidence)
                return intent, confidence
            except Exception as e:
                logger.log_error(e, "IntentClassifier.classify")

        # Keyword fallback
        intent, confidence = _keyword_classify(text)
        if intent:
            logger.log_intent(intent, confidence)
        return intent, confidence

    def should_use_rag(self, confidence: float) -> bool:
        return confidence < self.confidence_threshold

    def get_confidence_level(self, confidence: float) -> str:
        if confidence >= 0.7:
            return "high"
        elif confidence >= 0.5:
            return "medium"
        else:
            return "low"


intent_classifier = IntentClassifier()