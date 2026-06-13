"""
Food Domain Handler - Upgraded with Foodpanda/Google Maps links.
"""
import json
import os
import sqlite3
from typing import Tuple, Optional, Dict, Any, List
from domains.base_domain import BaseDomain
from nlp.entity_extractor import entity_extractor
from utils.logger import logger
from utils.helpers import get_project_root, fuzzy_match_choice, parse_distance, parse_price_range
from urllib.parse import quote as _q


class FoodDomain(BaseDomain):

    def __init__(self, context_manager):
        super().__init__("food", context_manager)
        self.food_data = self._load_food_data()
        self.db_path = os.path.join(get_project_root(), "data", "food.db")
        self.use_db = os.path.exists(self.db_path)

    def _load_food_data(self) -> dict:
        data_path = os.path.join(get_project_root(), "data", "food.json")
        try:
            if os.path.exists(data_path):
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.logger.info("Food data loaded from JSON file")
                    return data
        except Exception as e:
            logger.log_error(e, "FoodDomain._load_food_data")
        return {"food_places": {}, "food_items": [], "cities": []}

    def can_handle(self, intent: Optional[str], user_input: str) -> bool:
        text = user_input.lower()
        if intent == "food":
            return True
        if self.is_active():
            return True
        keywords = [
            "food", "eat", "hungry", "restaurant", "order", "delivery",
            "biryani", "burger", "pizza", "ice cream", "shake",
            "karahi", "nihari", "chinese", "bbq", "sundae",
            "dine", "khana", "khaana", "sasti", "sasta", "foodpanda"
        ]
        return any(k in text for k in keywords)

    def _get_food_links(self, food_item: str, city: str) -> str:
        q = _q(f"{food_item} in {city}")
        foodpanda = f"https://www.foodpanda.pk/restaurants/new?lat=33.6&lng=73.0&vertical=restaurants&search={_q(food_item)}"
        gmaps = f"https://www.google.com/maps/search/{q}"
        return f"🔗 Order on Foodpanda: {foodpanda}\n🔗 Find on Google Maps: {gmaps}"

    def handle(self, user_input: str, entities: Optional[Dict[str, Any]] = None) -> Tuple[str, Optional[str], List]:
        text_lower = user_input.lower().strip()
        if text_lower in ["order", "dine in", "dine", "delivery", "home delivery"]:
            if text_lower in ["order", "delivery", "home delivery"]:
                self.context_manager.set_entity("food_mode", "order")
            elif "dine" in text_lower:
                self.context_manager.set_entity("food_mode", "dine in")
            # Don't re-extract entities, just use existing context
            food_item = self.context_manager.get_entity("food_item")
            food_city = self.context_manager.get_entity("food_city")
            food_mode = self.context_manager.get_entity("food_mode")
            if food_item and food_city and food_mode:
                food_places = self.food_data.get("food_places", {})
                food_keys = list(food_places.keys())
                resolved = food_item if food_item.lower() in [k.lower() for k in food_keys] else (fuzzy_match_choice(food_item, food_keys, cutoff=0.45) or food_item)
                return self._get_restaurant_options(resolved or food_item, food_city, False)

        text = user_input.lower()
        asking_best = any(x in text for x in ["best", "top", "recommend", "link", "order online", "url"])

        if entities is None:
            entities = entity_extractor.extract_entities(text, domain="food")

        groq = self.context_manager.get_metadata("groq_enrichment") or {}
        groq_ents = groq.get("entities") or {}
        if groq_ents.get("location") and not entities.get("food_city"):
            entities["food_city"] = str(groq_ents["location"]).lower()
        if groq_ents.get("item") and not entities.get("food_item"):
            entities["food_item"] = str(groq_ents["item"]).lower()

        if entities.get("food_item"):
            self.context_manager.set_entity("food_item", entities["food_item"])

        new_city = entities.get("food_city") or entities.get("city")
        if new_city:
            self.context_manager.set_entity("food_city", new_city)
        if entities.get("user_location") and not self.context_manager.get_entity("food_city"):
            self.context_manager.set_entity("food_city", entities["user_location"])

        if any(x in text for x in ["dine", "dine in", "sit", "eat there"]):
            self.context_manager.set_entity("food_mode", "dine in")
        elif any(x in text for x in ["order", "delivery", "home", "deliver"]):
            self.context_manager.set_entity("food_mode", "order")

        if any(x in text for x in ["cheap", "cheapest", "budget", "sasta", "sasti"]):
            self.context_manager.set_entity("food_preference", "cheap")
        elif any(x in text for x in ["best", "top", "highest rating", "accha"]):
            self.context_manager.set_entity("food_preference", "best")

        food_item = self.context_manager.get_entity("food_item")
        food_city = self.context_manager.get_entity("food_city")
        food_mode = self.context_manager.get_entity("food_mode")

        if not food_item:
            food_items = self.food_data.get("food_items", [])
            suggestions = [c.title() for c in food_items[:6]]
            return "What would you like to eat? 🍽️", "food", suggestions

        if not food_city:
            cities = self.food_data.get("cities", ["Islamabad", "Rawalpindi", "Lahore", "Karachi"])
            suggestions = [c.title() for c in cities[:6]]
            return "Which city are you in? 📍", "food", suggestions

        if not food_mode:
            return "Do you want to dine in or order at home? 🏠", "food", ["Dine in", "Order"]

        food_places = self.food_data.get("food_places", {})
        food_keys = list(food_places.keys())
        resolved_food = food_item if food_item.lower() in [k.lower() for k in food_keys] else (
            fuzzy_match_choice(food_item, food_keys, cutoff=0.45) or food_item
        )

        return self._get_restaurant_options(resolved_food or food_item, food_city, asking_best)

    def _query_restaurants_db(self, food_type: str, city: str) -> List[Dict[str, Any]]:
        if not self.use_db or not os.path.exists(self.db_path):
            return []
        preference = self.context_manager.get_entity("food_preference") or ""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            if preference == "cheap":
                cur.execute("SELECT name, area, distance, price_range, rating FROM restaurants WHERE food_type=? AND city=? ORDER BY price_min ASC LIMIT 8", (food_type, city))
            elif preference == "best":
                cur.execute("SELECT name, area, distance, price_range, rating FROM restaurants WHERE food_type=? AND city=? ORDER BY rating DESC LIMIT 8", (food_type, city))
            else:
                cur.execute("SELECT name, area, distance, price_range, rating FROM restaurants WHERE food_type=? AND city=? ORDER BY distance ASC, rating DESC LIMIT 8", (food_type, city))
            rows = cur.fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.log_error(e, "FoodDomain._query_restaurants_db")
            return []

    def _get_restaurant_options(self, food_item: str, city: str, asking_best: bool = False) -> Tuple[str, Optional[str], List]:
        food_places = self.food_data.get("food_places", {})
        food_key = food_item if food_item in food_places else (
            fuzzy_match_choice(food_item, list(food_places.keys()), cutoff=0.45) or food_item
        )
        cities_for_food = list(food_places.get(food_key, {}).keys())
        city_lower = city.strip().lower()
        city_key = city_lower if city_lower in cities_for_food else (
            fuzzy_match_choice(city, cities_for_food, cutoff=0.45) or city_lower
        )

        places = self._query_restaurants_db(food_key, city_key) if self.use_db else []
        if not places:
            places = food_places.get(food_key, {}).get(city_key, [])

        links = self._get_food_links(food_item, city)

        if not places:
            self.context_manager.reset()
            return (
                f"Sorry, no {food_key or food_item} options found in {city.title()}.\n\n"
                f"Try ordering online:\n{links}",
                None, []
            )

        preference = self.context_manager.get_entity("food_preference") or ""
        if preference == "cheap":
            places = sorted(places, key=lambda p: parse_price_range(p.get("price_range", "")) if isinstance(p, dict) else float("inf"))
        elif preference == "best":
            places = sorted(places, key=lambda p: -float(p.get("rating", 0)) if isinstance(p, dict) else 0)
        else:
            places = sorted(places, key=lambda p: (
                parse_distance(p.get("distance", "")) if isinstance(p, dict) else float("inf"),
                -float(p.get("rating", 0)) if isinstance(p, dict) and p.get("rating") else 0
            ))

        places = places[:8]

        # Step 3 — best
        if asking_best:
            top = places[0]
            if isinstance(top, dict):
                name = top.get("name", "")
                rating = top.get("rating", "")
                area = top.get("area", "")
                price = top.get("price_range", "")
                self.context_manager.reset()
                return (
                    f"🏆 Best {food_item.title()} in {city.title()}:\n\n"
                    f"🍕 {name}\n"
                    f"📍 {area}\n"
                    f"💰 {price}\n"
                    f"⭐ {rating}\n\n"
                    f"Order online:\n{links}",
                    None, []
                )

        formatted = []
        for place in places:
            if isinstance(place, dict):
                rating = f" ⭐{place['rating']}" if place.get('rating') else ""
                line = f"🍕 {place.get('name', '')}"
                if place.get('distance') and place.get('area'):
                    line += f" - {place['distance']} from {place['area']}"
                line += rating
                formatted.append(line)
            else:
                formatted.append(f"🍕 {place}")

        logger.logger.info(f"GROUNDING: food_item={food_key}, city={city_key}, count={len(places)}")
        self.context_manager.reset()

        result = "\n".join(formatted)
        result += f"\n\nOrder online:\n{links}"
        return result, None, []
