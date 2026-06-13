"""
E-Commerce Domain Handler - Upgraded with Daraz/PriceOye links and budget filtering.
"""
import json
import os
import re
from typing import Tuple, Optional, Dict, Any
from domains.base_domain import BaseDomain
from nlp.entity_extractor import entity_extractor
from utils.logger import logger
from utils.helpers import get_project_root
from urllib.parse import quote as _q


def _parse_price(price_str: str) -> Optional[float]:
    if not price_str:
        return None
    try:
        cleaned = re.sub(r'[^\d.]', '', str(price_str).replace(',', ''))
        nums = re.findall(r'\d+', cleaned)
        if nums:
            return float(nums[0])
    except Exception:
        pass
    return None


class ProductsDomain(BaseDomain):

    def __init__(self, context_manager):
        super().__init__("products", context_manager)
        self.products_data = self._load_products_data()

    def _load_products_data(self) -> dict:
        data_path = os.path.join(get_project_root(), "data", "products.json")
        try:
            if os.path.exists(data_path):
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.logger.info("Products data loaded from JSON file")
                    return data
        except Exception as e:
            logger.log_error(e, "ProductsDomain._load_products_data")
        return {"products": {}, "shopping_sites": []}

    def can_handle(self, intent: Optional[str], user_input: str) -> bool:
        text = user_input.lower()
        if intent in ("products", "ecommerce", "price_filter"):
            return True
        if self.is_active():
            return True
        keywords = [
            "buy", "shop", "purchase", "order", "mobile", "laptop",
            "headphones", "watch", "product", "price", "tablet",
            "camera", "tv", "speaker", "phone", "daraz", "priceoye",
            "amazon", "online shopping"
        ]
        return any(k in text for k in keywords)

    PRODUCT_ALIASES = {
        "phone": "mobile", "phones": "mobile", "notebook": "laptop",
        "headphone": "headphones", "watches": "watch", "mobiles": "mobile",
        "tablets": "tablet", "television": "tv", "televisions": "tv",
        "speakers": "speaker", "cameras": "camera"
    }

    def _get_shopping_links(self, product: str) -> str:
        q = _q(product)
        daraz = f"https://www.daraz.pk/catalog/?q={q}"
        priceoye = f"https://priceoye.pk/search?q={q}"
        amazon = f"https://www.amazon.com/s?k={q}"
        olx = f"https://www.olx.com.pk/q/{q}"
        return f"🔗 Daraz: {daraz}\n🔗 PriceOye: {priceoye}\n🔗 Amazon: {amazon}\n🔗 OLX: {olx}"

    def handle(self, user_input: str, entities: Optional[Dict[str, Any]] = None) -> Tuple[str, Optional[str], list]:
        text = user_input.lower()
        asking_best = any(x in text for x in ["best", "top", "which one", "recommend", "link", "buy now", "url", "site"])

        if entities is None:
            entities = entity_extractor.extract_entities(text, domain="ecommerce")
            if not entities.get("ecommerce_product"):
                entities = entity_extractor.extract_entities(text, domain="products")

        budget = entities.get("budget") or self.context_manager.get_entity("budget")
        if budget:
            try:
                budget = float(budget)
                self.context_manager.set_entity("budget", budget)
            except Exception:
                budget = None

        price_range = entities.get("price_range") or self.context_manager.get_entity("price_range")

        product_category = (
            entities.get("ecommerce_product") or
            entities.get("product") or
            self.context_manager.get_entity("product_category")
        )
        if product_category:
            product_category = str(product_category).lower().strip()
            product_category = self.PRODUCT_ALIASES.get(product_category, product_category)
            self.context_manager.set_entity("product_category", product_category)

        if product_category:
            products_data = self.products_data.get("products", {})
            if product_category in products_data:
                product_list = products_data[product_category]

                # Budget filter
                if budget:
                    filtered = [p for p in product_list if isinstance(p, dict) and (
                        _parse_price(str(p.get("price", ""))) or 999999999
                    ) <= budget]
                    if filtered:
                        product_list = filtered
                elif price_range == "budget":
                    product_list = sorted(
                        product_list,
                        key=lambda p: _parse_price(str(p.get("price", ""))) or 999999 if isinstance(p, dict) else 999999
                    )

                product_list = product_list[:8]
                links = self._get_shopping_links(product_category)

                # Step 3 — best
                if asking_best and product_list:
                    top = product_list[0]
                    name = top.get("name", "")
                    price = top.get("price", "")
                    retailer = top.get("retailer", "")
                    url = top.get("url", "")
                    self.context_manager.reset()
                    response = f"🏆 Best {product_category.title()} for you:\n\n"
                    response += f"🛍️ {name}\n"
                    response += f"💰 {price}\n"
                    response += f"🏪 {retailer}\n"
                    if url:
                        response += f"🔗 {url}\n"
                    response += f"\nMore options:\n{links}"
                    return response, None, []

                # Step 1 & 2 — show all
                formatted = []
                for p in product_list:
                    if isinstance(p, dict):
                        name = p.get("name", "")
                        price = p.get("price", "")
                        retailer = p.get("retailer", "")
                        avail = p.get("availability", "")
                        url = p.get("url", "")
                        line = f"🛍️ {name} - {price} @ {retailer} ({avail})"
                        if url:
                            line += f"\n   🔗 {url}"
                        formatted.append(line)
                    else:
                        formatted.append(f"🛍️ {p}")

                self.context_manager.reset()
                return "\n".join(formatted) + f"\n\nSearch more:\n{links}", None, []

            else:
                self.context_manager.reset()
                return f"Sorry, no {product_category} products found.", None, []

        # Fallback
        categories = list(self.products_data.get("products", {}).keys())
        suggestions = [p.title() for p in categories[:6]]
        return f"What do you want to shop for?\n{', '.join(suggestions)}?", "products", suggestions
