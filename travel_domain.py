"""
Travel Domain Handler - Upgraded with Booking/Sastaticket links and 3-step flow.
"""
import json
import os
from typing import Tuple, Optional, Dict, Any
from domains.base_domain import BaseDomain
from nlp.entity_extractor import entity_extractor
from utils.logger import logger
from utils.helpers import get_project_root


class TravelDomain(BaseDomain):

    def __init__(self, context_manager):
        super().__init__("travel", context_manager)
        self.trips_data = self._load_data()

    def _load_data(self) -> dict:
        path = os.path.join(get_project_root(), "data", "trips.json")
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.logger.info("Travel trips data loaded from JSON file")
                    return data
        except Exception as e:
            logger.log_error(e, "TravelDomain._load_data")
        return {"trips": {}}

    def can_handle(self, intent: Optional[str], user_input: str) -> bool:
        text = user_input.lower()
        if intent == "travel":
            return True
        if self.is_active():
            return True
        keywords = [
            "travel", "trip", "tour", "visit", "flight", "hotel",
            "vacation", "holiday", "booking", "hunza", "swat", "murree",
            "naran", "skardu", "nathia", "abbottabad", "sastaticket",
            "bookme", "ticket"
        ]
        return any(k in text for k in keywords)

    def _get_booking_links(self, place: str) -> str:
        q = place.replace(" ", "+")
        booking = f"https://www.booking.com/searchresults.html?ss={q}+Pakistan"
        sastaticket = f"https://www.sastaticket.pk/hotels/{q.lower()}"
        bookme = f"https://bookme.pk/hotels?city={q}"
        return f"🔗 Booking.com: {booking}\n🔗 Sastaticket: {sastaticket}\n🔗 Bookme: {bookme}"

    def handle(self, user_input: str, entities: Optional[Dict[str, Any]] = None) -> Tuple[str, Optional[str], list]:
        text = user_input.lower()
        asking_best = any(x in text for x in ["best", "top", "recommend", "link", "book", "url", "ticket"])

        if entities is None:
            entities = entity_extractor.extract_entities(text, domain="travel")

        if entities.get("trip_place"):
            self.context_manager.set_entity("trip_place", entities["trip_place"])

        trip_place = self.context_manager.get_entity("trip_place") or entities.get("trip_place")

        trips = self.trips_data.get("trips", {})

        if trip_place:
            # Find trip data
            trip_key = trip_place.lower()
            trip_info = None
            for key, val in trips.items():
                if trip_place.lower() in key.lower() or key.lower() in trip_place.lower():
                    trip_key = key
                    trip_info = val
                    break

            if trip_info:
                links = self._get_booking_links(trip_place)

                if asking_best:
                    self.context_manager.reset()
                    return (
                        f"🏆 Best choice: {trip_key.title()}\n\n"
                        f"📍 {trip_info.get('about', '')[:100]}\n"
                        f"📅 Best time: {trip_info.get('best_time', '')}\n\n"
                        f"Book now:\n{links}",
                        None, []
                    )

                # Format trip details
                response = f"🧳 {trip_key.title()}\n"
                response += f"{trip_info.get('about', '')}\n"
                response += f"Best time: {trip_info.get('best_time', '')}\n\n"

                hotels = trip_info.get("hotels", [])
                if hotels:
                    response += "Hotels:\n"
                    for h in hotels[:3]:
                        if isinstance(h, dict):
                            response += f"  🏨 {h.get('name', '')} - {h.get('price', '')}\n"

                response += f"\nBook online:\n{links}"
                self.context_manager.reset()
                return response, None, []

            else:
                links = self._get_booking_links(trip_place)
                self.context_manager.reset()
                return (
                    f"✈️ Planning a trip to {trip_place.title()}?\n\n"
                    f"Book here:\n{links}",
                    None, []
                )

        # Ask for destination
        popular = ["Hunza", "Swat", "Murree", "Nathia Gali", "Skardu", "Naran"]
        return (
            f"✈️ Where would you like to travel?\n"
            f"Popular: {', '.join(popular)}",
            "travel", popular
        )
