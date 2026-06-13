"""
Automobiles Domain Handler - Upgraded with PakWheels/OLX links and 3-step flow.
"""
import json
import os
import re
from typing import Tuple, Optional, Dict, Any, List
from domains.base_domain import BaseDomain
from nlp.entity_extractor import entity_extractor
from utils.logger import logger
from utils.helpers import get_project_root, fuzzy_match_choice


def _parse_lac_price(price_str: str) -> Optional[float]:
    """Parse price like 'PKR 45 - 60 lacs' → returns lower bound in rupees."""
    if not price_str:
        return None
    try:
        nums = re.findall(r'[\d.]+', str(price_str))
        if nums:
            val = float(nums[0])
            if val < 1000:
                val = val * 100000
            return val
    except Exception:
        pass
    return None


class AutomobilesDomain(BaseDomain):

    def __init__(self, context_manager):
        super().__init__("automobiles", context_manager)
        self.autos_data = self._load_data()

    def _load_data(self) -> dict:
        path = os.path.join(get_project_root(), "data", "automobiles.json")
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.log_error(e, "AutomobilesDomain._load_data")
        return {"cars": {}}

    def can_handle(self, intent: Optional[str], user_input: str) -> bool:
        text = user_input.lower()
        if intent == "automobiles":
            return True
        if self.is_active():
            return True
        keywords = [
            "car", "cars", "vehicle", "auto", "bike", "sedan", "suv",
            "hatchback", "pickup", "electric car", "gaari", "gari", "gaadi",
            "honda", "toyota", "suzuki", "kia", "hyundai", "changan",
            "pakwheels", "olx car", "buy car", "rent car"
        ]
        return any(k in text for k in keywords)

    def handle(self, user_input: str, entities: Optional[Dict[str, Any]] = None) -> Tuple[str, Optional[str], list]:
        text = user_input.lower()
        asking_best = any(x in text for x in ["best", "top", "which one", "recommend", "link", "buy now", "url", "site"])

        if entities is None:
            entities = entity_extractor.extract_entities(text, domain="automobiles")

        # Update context
        if entities.get("auto_type"):
            self.context_manager.set_entity("auto_type", entities["auto_type"])
        if entities.get("auto_brand"):
            self.context_manager.set_entity("auto_brand", entities["auto_brand"])
        if entities.get("budget"):
            self.context_manager.set_entity("budget", entities["budget"])

        auto_type = self.context_manager.get_entity("auto_type") or entities.get("auto_type")
        budget = self.context_manager.get_entity("budget") or entities.get("budget")
        brand = self.context_manager.get_entity("auto_brand") or entities.get("auto_brand")

        cars_data = self.autos_data.get("cars", {})

        if auto_type:
            car_keys = list(cars_data.keys())
            resolved = auto_type if auto_type in car_keys else (
                fuzzy_match_choice(auto_type, car_keys, cutoff=0.45) or auto_type
            )
            car_type_data = cars_data.get(resolved, {})
            models = car_type_data.get("models", [])

            # Filter by budget
            if budget:
                try:
                    budget_float = float(budget)
                    filtered = [m for m in models if isinstance(m, dict) and (
                        _parse_lac_price(m.get("price", "")) or 999999999
                    ) <= budget_float]
                    if filtered:
                        models = filtered
                except Exception:
                    pass

            # Filter by brand
            if brand:
                brand_filtered = [m for m in models if isinstance(m, dict) and
                                  brand.lower() in m.get("name", "").lower()]
                if brand_filtered:
                    models = brand_filtered

            if models:
                # Step 3 — best
                if asking_best:
                    top = models[0]
                    name = top.get("name", "")
                    price = top.get("price", "")
                    location = top.get("dealership_location", "")
                    url = top.get("url", "")
                    olx_url = f"https://www.olx.com.pk/cars/{name.lower().replace(' ', '-')}/"

                    response = f"🏆 Best {resolved.title()} for you:\n\n"
                    response += f"🚗 {name}\n"
                    response += f"💰 {price}\n"
                    response += f"📍 {location}\n"
                    if url:
                        response += f"🔗 PakWheels: {url}\n"
                    response += f"🔗 OLX: {olx_url}"
                    self.context_manager.reset()
                    return response, None, []

                # Step 1 & 2 — show all
                formatted = []
                for m in models[:6]:
                    if isinstance(m, dict):
                        name = m.get("name", "")
                        price = m.get("price", "")
                        avail = m.get("availability", "")
                        url = m.get("url", "")
                        line = f"🚗 {name} - {price} ({avail})"
                        if url:
                            line += f"\n   🔗 {url}"
                        formatted.append(line)

                desc = car_type_data.get("description", "")
                self.context_manager.reset()
                return f"🚗 {resolved.title()} options:\n\n" + "\n".join(formatted), None, []

        # Ask for car type
        types = list(cars_data.keys())
        suggestions = [t.title() for t in types]
        return "What type of car are you looking for?\nSedan, SUV, Hatchback, Electric, or Pickup?", "automobiles", suggestions
