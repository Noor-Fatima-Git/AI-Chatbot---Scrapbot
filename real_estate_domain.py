"""
Real Estate Domain Handler - Upgraded with Zameen/OLX/Graana links and 3-step flow.
"""
import json
import os
from typing import Tuple, Optional, Dict, Any, List
from domains.base_domain import BaseDomain
from utils.logger import logger
from utils.helpers import get_project_root


class RealEstateDomain(BaseDomain):

    def __init__(self, context_manager):
        super().__init__("real_estate", context_manager)
        self.listings = self._load_data()

    def _load_data(self) -> dict:
        path = os.path.join(get_project_root(), "data", "real_estate.json")
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.log_error(e, "RealEstateDomain._load_data")
        return {"real_estate": {}}

    def can_handle(self, intent: Optional[str], user_input: str) -> bool:
        text = user_input.lower()
        if intent == "real_estate":
            return True
        if self.is_active():
            return True
        keywords = [
            "rent", "rental", "flat", "house", "property", "real estate",
            "apartment", "plot", "home", "villa", "studio", "bedroom",
            "zameen", "graana", "buy house", "lease", "ghar", "makaan",
            "makan", "marla", "kanal"
        ]
        return any(k in text for k in keywords)

    def _detect_city(self, text: str) -> Optional[str]:
        text = text.lower()
        for city in ["islamabad", "lahore", "karachi", "rawalpindi", "peshawar", "quetta"]:
            if city in text:
                return city
        loc = self.context_manager.get_entity("user_location")
        if isinstance(loc, str):
            for city in ["islamabad", "lahore", "karachi", "rawalpindi"]:
                if city in loc.lower():
                    return city
        return None

    def _get_links(self, city: str, mode: str, prop_type: str = "") -> str:
        q = f"{prop_type}+{mode}+{city}".replace(" ", "+")
        zameen = f"https://www.zameen.com/Homes/{city.title()}-1-1.html"
        olx = f"https://www.olx.com.pk/properties_{mode}/"
        graana = f"https://www.graana.com/search?location={city}&purpose={mode}"
        return f"🔗 Zameen: {zameen}\n🔗 OLX: {olx}\n🔗 Graana: {graana}"

    def handle(self, user_input: str, entities: Optional[Dict[str, Any]] = None) -> Tuple[str, Optional[str], list]:
        text = user_input.lower()
        asking_best = any(x in text for x in ["best", "top", "which one", "recommend", "link", "url", "site"])

        city = self._detect_city(text)
        if city:
            self.context_manager.set_entity("user_location", city)

        mode = "rent"
        if any(x in text for x in ["buy", "purchase", "sale", "khareed"]):
            mode = "buy"
        elif any(x in text for x in ["rent", "rental", "lease", "kiraya"]):
            mode = "rent"

        city = city or self.context_manager.get_entity("user_location")

        if not city:
            return (
                "🏠 Let's find you a property!\nWhich city are you interested in?",
                "real_estate",
                ["Islamabad", "Lahore", "Karachi", "Rawalpindi"]
            )

        listings = self.listings.get("real_estate", {}).get(city, [])

        if not listings:
            links = self._get_links(city, mode)
            self.context_manager.reset()
            return (
                f"🏠 No listings found locally for {city.title()}.\n\n"
                f"Search on these platforms:\n{links}",
                None, []
            )

        # Filter by mode
        mode_filtered = [l for l in listings if isinstance(l, dict) and l.get("mode") == mode]
        if mode_filtered:
            listings = mode_filtered

        # Step 3 — best
        if asking_best:
            top = listings[0]
            title = top.get("title", "")
            price = top.get("price", "")
            area = top.get("area", "")
            links = self._get_links(city, mode, top.get("type", ""))
            self.context_manager.reset()
            return (
                f"🏆 Best {mode} option in {city.title()}:\n\n"
                f"🏠 {title}\n"
                f"💰 PKR {price}\n"
                f"📍 {area}, {city.title()}\n\n"
                f"Search more:\n{links}",
                None, []
            )

        # Step 1 & 2 — show all
        formatted = []
        for l in listings[:6]:
            if isinstance(l, dict):
                title = l.get("title", "")
                price = l.get("price", "")
                area = l.get("area", "")
                formatted.append(f"🏠 {title} - PKR {price} ({area})")

        links = self._get_links(city, mode)
        self.context_manager.reset()
        return (
            f"🏠 {mode.title()} options in {city.title()}:\n\n" +
            "\n".join(formatted) +
            f"\n\nSearch more:\n{links}",
            None, []
        )
