"""
Upgraded Entity Extraction Module for ScrapBot.
Handles typos, budget extraction, fuzzy matching, Roman Urdu.
"""

import re
from typing import Dict, List, Optional, Any

try:
    import spacy
except Exception:
    spacy = None

from utils.logger import logger
from utils.synonyms import resolve_entities as resolve_entity_synonyms


# ── TYPO CORRECTION DICTIONARY ─────────────────────────────────────
TYPO_MAP = {
    # phones/products
    "fone": "phone", "phon": "phone", "moble": "mobile", "mobil": "mobile",
    "labtop": "laptop", "laptoop": "laptop", "laptp": "laptop",
    "headfone": "headphones", "headfons": "headphones",
    "camra": "camera", "camrea": "camera",
    "tablat": "tablet", "tablt": "tablet",
    "watche": "watch", "wach": "watch",
    # jobs
    "enginer": "engineer", "engneer": "engineer", "enginear": "engineer",
    "softwar": "software", "sofware": "software", "softwre": "software",
    "develper": "developer", "devloper": "developer", "develope": "developer",
    "manger": "manager", "managr": "manager",
    "acountant": "accountant", "accountnt": "accountant",
    "teachr": "teacher", "techer": "teacher",
    "desiner": "designer", "disigner": "designer",
    "analst": "analyst", "anlayst": "analyst",
    # food
    "biyrani": "biryani", "biriyani": "biryani", "biryni": "biryani",
    "burgr": "burger", "brger": "burger",
    "piza": "pizza", "pizzza": "pizza",
    "karahi": "karahi", "kraahi": "karahi",
    # cities
    "lahor": "lahore", "lahre": "lahore",
    "karachy": "karachi", "krachi": "karachi",
    "islamabd": "islamabad", "islmabad": "islamabad",
    "rawalindi": "rawalpindi", "rwalpindi": "rawalpindi",
    # price/budget
    "cheep": "cheap", "chep": "cheap", "cheapp": "cheap",
    "expnsive": "expensive", "expnsiv": "expensive",
    "afrdable": "affordable", "afodable": "affordable",
    # common
    "nede": "need", "waant": "want", "wnat": "want",
    "recomend": "recommend", "reccomend": "recommend",
    "srch": "search", "serch": "search",
}

# ── ROMAN URDU → ENGLISH ────────────────────────────────────────────
ROMAN_URDU_MAP = {
    "mujhe": "i want", "chahiye": "i need", "chahie": "i need",
    "sasta": "cheap", "sasti": "cheap", "sastay": "cheap",
    "mehnga": "expensive", "mehenga": "expensive",
    "kaam": "job", "nokri": "job", "naukri": "job",
    "ghar": "house", "makaan": "house", "makan": "house",
    "gaari": "car", "gari": "car", "gaadi": "car",
    "khana": "food", "khaana": "food",
    "phone": "mobile", "mobile": "mobile",
    "accha": "good", "acha": "good", "achha": "good",
    "bura": "bad", "bura": "bad",
    "mein": "in", "main": "in",
    "ka": "", "ki": "", "ke": "", "ko": "",
    "aur": "and", "or": "and",
    "kitna": "how much", "kitni": "how much",
    "kahan": "where", "kaha": "where",
    "wala": "", "wali": "", "walay": "",
}

# ── BUDGET PATTERNS ─────────────────────────────────────────────────
BUDGET_PATTERNS = [
    (r'under\s+(\d[\d,]*)\s*k', lambda m: int(m.group(1)) * 1000),
    (r'under\s+(\d[\d,]*)\s*lac', lambda m: int(m.group(1)) * 100000),  # ← ADD THIS
    (r'under\s+(\d[\d,]*)', lambda m: int(m.group(1).replace(',', ''))),
    (r'below\s+(\d[\d,]*)\s*k', lambda m: int(m.group(1)) * 1000),
    (r'below\s+(\d[\d,]*)\s*lac', lambda m: int(m.group(1)) * 100000),  # ← ADD THIS
    (r'below\s+(\d[\d,]*)', lambda m: int(m.group(1).replace(',', ''))),
    (r'less\s+than\s+(\d[\d,]*)\s*k', lambda m: int(m.group(1)) * 1000),
    (r'less\s+than\s+(\d[\d,]*)', lambda m: int(m.group(1).replace(',', ''))),
    (r'budget\s+(\d[\d,]*)\s*k', lambda m: int(m.group(1)) * 1000),
    (r'(\d[\d,]*)\s*k\s+budget', lambda m: int(m.group(1)) * 1000),
    (r'(\d[\d,]*)\s*k\s+ka', lambda m: int(m.group(1)) * 1000),
    (r'(\d[\d,]*)\s*k', lambda m: int(m.group(1)) * 1000),
    (r'(\d[\d,]*)\s*lac', lambda m: int(m.group(1)) * 100000),          # ← ADD THIS
    (r'pkr\s*(\d[\d,]*)', lambda m: int(m.group(1).replace(',', ''))),
    (r'rs\.?\s*(\d[\d,]*)', lambda m: int(m.group(1).replace(',', ''))),
    (r'(\d[\d,]{4,})', lambda m: int(m.group(1).replace(',', ''))),
]

def correct_typos(text: str) -> str:
    """Fix common typos and Roman Urdu words."""
    words = text.lower().split()
    corrected = []
    for word in words:
        clean = re.sub(r'[^\w]', '', word)
        if clean in TYPO_MAP:
            corrected.append(TYPO_MAP[clean])
        elif clean in ROMAN_URDU_MAP:
            val = ROMAN_URDU_MAP[clean]
            if val:
                corrected.append(val)
        else:
            corrected.append(word)
    return ' '.join(corrected)


def extract_budget(text: str) -> Optional[int]:
    """Extract budget/price limit from text."""
    for pattern, extractor in BUDGET_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return extractor(match)
            except Exception:
                pass
    return None


def extract_price_range(text: str) -> Optional[str]:
    """Extract price range label."""
    if any(w in text for w in ['cheap', 'sasta', 'sasti', 'budget', 'affordable', 'low price', 'low cost']):
        return 'budget'
    if any(w in text for w in ['expensive', 'premium', 'luxury', 'high end', 'best']):
        return 'premium'
    if any(w in text for w in ['mid range', 'medium', 'moderate', 'average']):
        return 'mid'
    return None


class EntityExtractor:
    """
    Upgraded entity extractor with typo correction, budget extraction,
    fuzzy matching and Roman Urdu support.
    """

    def __init__(self):
        self.nlp = None
        if spacy is not None:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.logger.info("spaCy model loaded successfully")
            except OSError:
                logger.logger.warning(
                    "spaCy model 'en_core_web_sm' not found. "
                    "Install with: python -m spacy download en_core_web_sm. "
                    "Falling back to pattern-based extraction."
                )
            except Exception as e:
                logger.log_error(e, "EntityExtractor.__init__")

    def extract_entities(self, text: str, domain: Optional[str] = None) -> Dict[str, Any]:
        """Extract all entities with typo correction and budget extraction."""
        entities = {}

        # Step 1: correct typos and Roman Urdu
        corrected = correct_typos(text)
        text_lower = corrected.lower()

        # Step 2: extract budget
        budget = extract_budget(text_lower)
        if budget:
            entities["budget"] = budget

        price_range = extract_price_range(text_lower)
        if price_range:
            entities["price_range"] = price_range

        # Step 3: spaCy NER
        if self.nlp:
            doc = self.nlp(text)
            cities = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
            if cities:
                entities["cities"] = cities
            dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
            if dates:
                entities["dates"] = dates

        # Step 4: domain-specific extraction on corrected text
        entities.update(self._extract_food_entities(text_lower))
        entities.update(self._extract_trip_entities(text_lower))
        entities.update(self._extract_ecommerce_entities(text_lower))
        entities.update(self._extract_job_entities(text_lower))
        entities.update(self._extract_automobile_entities(text_lower))
        entities.update(self._extract_flight_entities(text_lower))

        # Step 5: city extraction
        if "cities" not in entities:
            city = self._extract_city(text_lower)
            if city:
                entities["city"] = city

        # Step 6: user location
        loc = self._extract_user_location(text_lower, entities, domain)
        if loc:
            entities["user_location"] = loc

        # Step 7: date extraction
        if "dates" not in entities:
            date = self._extract_date(text_lower)
            if date:
                entities["date"] = date

        # Step 8: resolve synonyms
        entities = resolve_entity_synonyms(entities)

        if entities:
            logger.log_entities(entities)

        return entities

    def _extract_food_entities(self, text: str) -> Dict[str, Optional[str]]:
        foods = [
            "biryani", "burger", "pizza", "sundae", "ice cream",
            "shake", "dessert", "karahi", "nihari", "chinese", "bbq",
            "food", "restaurant", "eat", "dine", "delivery"
        ]
        cities = ["islamabad", "rawalpindi", "karachi", "lahore", "multan", "peshawar"]
        food = next((f for f in foods if f in text), None)
        city = next((c for c in cities if c in text), None)
        result = {}
        if food:
            result["food_item"] = food
        if city:
            result["food_city"] = city
        return result

    def _extract_trip_entities(self, text: str) -> Dict[str, Any]:
        trip_places = [
            "hunza", "swat", "murree", "nathia gali",
            "shogran", "abbottabad", "skardu", "gilgit", "naran"
        ]
        place = next((p for p in trip_places if p in text), None)
        want_hotel = bool(re.search(r"hotel|stay|room|booking|book", text))
        want_time = bool(re.search(r"best time|when|season|weather", text))
        result = {}
        if place:
            result["trip_place"] = place
        if want_hotel:
            result["trip_hotel"] = True
        if want_time:
            result["trip_time"] = True
        return result

    def _extract_ecommerce_entities(self, text: str) -> Dict[str, Any]:
        products = [
            "mobile", "phone", "laptop", "headphones", "watch",
            "tablet", "camera", "tv", "speaker", "computer",
            "earphones", "earbuds", "smartwatch", "gaming"
        ]
        product = next((p for p in products if p in text), None)
        want_price = bool(re.search(r"price|cost|range|how much|kitna", text))
        want_buy = bool(re.search(r"buy|order|purchase|shop|khareed", text))
        result = {}
        if product:
            result["ecommerce_product"] = product
        if want_price:
            result["ecommerce_price"] = True
        if want_buy:
            result["ecommerce_buy"] = True
        return result

    def _extract_job_entities(self, text: str) -> Dict[str, Optional[str]]:
        job_titles = [
            "software engineer", "software developer", "web developer",
            "teacher", "accountant", "graphic designer", "data analyst",
            "developer", "manager", "doctor", "nurse", "marketing manager",
            "hr manager", "content writer", "sales executive",
            "mechanical engineer", "electrical engineer", "data scientist",
            "product manager", "project manager", "business analyst",
            "full stack", "frontend", "backend", "devops", "mobile developer",
        ]
        cities = ["islamabad", "rawalpindi", "lahore", "karachi", "multan", "peshawar", "faisalabad"]
        job = next((j for j in job_titles if j in text), None)
        city = next((c for c in cities if c in text), None)

        # employment type
        emp_type = None
        if any(w in text for w in ["remote", "work from home", "wfh", "online"]):
            emp_type = "remote"
        elif any(w in text for w in ["part time", "part-time", "parttime"]):
            emp_type = "part-time"
        elif any(w in text for w in ["full time", "full-time", "fulltime"]):
            emp_type = "full-time"
        elif any(w in text for w in ["internship", "intern", "trainee"]):
            emp_type = "internship"

        result = {}
        if job:
            result["job_title"] = job
        if city:
            result["job_city"] = city
        if emp_type:
            result["employment_type"] = emp_type
        return result

    def _extract_automobile_entities(self, text: str) -> Dict[str, Optional[str]]:
        car_types = ["sedan", "suv", "hatchback", "electric", "hybrid", "pickup", "truck", "van"]
        fuels = ["petrol", "diesel", "hybrid", "electric", "cng"]
        brands = [
            "toyota", "honda", "suzuki", "kia", "hyundai", "mg",
            "changan", "haval", "united", "daehan", "isuzu", "ford",
            "nissan", "mitsubishi", "audi", "bmw", "mercedes", "tesla"
        ]
        car_type = next((t for t in car_types if t in text), None)
        fuel = next((f for f in fuels if f in text), None)
        brand = next((b for b in brands if b in text), None)
        result = {}
        if car_type:
            result["auto_type"] = car_type
        if fuel:
            result["auto_fuel"] = fuel
        if brand:
            result["auto_brand"] = brand
        return result

    def _extract_flight_entities(self, text: str) -> Dict[str, Any]:
        cities = ["karachi", "islamabad", "lahore", "multan", "peshawar", "quetta", "faisalabad"]
        city_map = {"pta": "peshawar", "isb": "islamabad", "khi": "karachi", "lhr": "lahore"}
        for abbr, full in city_map.items():
            if abbr in text:
                text = text.replace(abbr, full)
        found = [c for c in cities if c in text]
        source = found[0] if found else None
        dest = found[1] if len(found) > 1 else None
        flight_class = next((c for c in ["economy", "business", "premium"] if c in text), None)
        result = {}
        if source:
            result["flight_source"] = source
        if dest:
            result["flight_dest"] = dest
        if flight_class:
            result["flight_class"] = flight_class
        return result

    def _extract_user_location(self, text: str, entities: Dict[str, Any], domain: Optional[str]) -> Optional[str]:
        if entities.get("food_city"):
            return entities["food_city"]
        if entities.get("job_city"):
            return entities["job_city"]
        if entities.get("city"):
            return entities["city"]
        if isinstance(entities.get("cities"), list) and entities["cities"]:
            return entities["cities"][0].lower()
        areas = ["blue area", "f-10", "f-11", "f-7", "f-6", "g-10", "dha", "bahria", "gulberg", "saddar"]
        return next((a for a in areas if a in text), None)

    def _extract_city(self, text: str) -> Optional[str]:
        cities = ["islamabad", "rawalpindi", "karachi", "lahore", "multan", "peshawar", "quetta", "faisalabad"]
        return next((c for c in cities if c in text), None)

    def _extract_date(self, text: str) -> Optional[str]:
        patterns = {"today": "today", "tomorrow": "tomorrow", "next week": "next week", "next month": "next month"}
        for pattern, value in patterns.items():
            if pattern in text:
                return value
        return None


# singleton
entity_extractor = EntityExtractor()