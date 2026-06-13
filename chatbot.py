"""
Main Chatbot Orchestration Module for Scrapbot.

This module coordinates all components of the chatbot:
- Intent classification
- Entity extraction
- Domain routing
- Context management
- RAG fallback

The chatbot follows this flow:
1. User input → Normalize text
2. Intent classification → Determine user intent
3. Entity extraction → Extract structured information
4. Domain routing → Route to appropriate domain handler
5. Domain processing → Generate response using domain logic
6. RAG fallback → If intent confidence is low or knowledge question

This architecture enables:
- Modular design: Each component is independent
- Easy extension: Add new domains without changing core logic
- Context awareness: Multi-turn conversations
- Intelligent fallback: RAG for knowledge questions
"""

from typing import Optional, Dict, Any, Tuple, List
from context.context_manager import context_manager

# Per-session conversation history for Groq enrichment (last 10 turns)
conversation_sessions: Dict[str, List[Dict[str, str]]] = {}

# Map Groq enrichment domain labels to chatbot domain keys
_GROQ_DOMAIN_TO_HANDLER = {
    "food": "food",
    "travel": "travel",
    "ecommerce": "products",
    "jobs": "jobs",
    "automobiles": "automobiles",
    "unknown": None,
}

# Map Groq domain labels to intent strings expected by domain handlers
_GROQ_DOMAIN_TO_INTENT = {
    "food": "food",
    "travel": "travel",
    "ecommerce": "ecommerce",
    "jobs": "job",
    "automobiles": None,
}

GROQ_DOMAIN_CONFIDENCE_THRESHOLD = 0.6
GROQ_DISAGREEMENT_CONFIDENCE_THRESHOLD = 0.8

# Confidence bands: high = proceed with intent; medium = proceed but log; low = prefer clarification/RAG.
CONFIDENCE_HIGH = 0.7
CONFIDENCE_MEDIUM = 0.5
CONFIDENCE_THRESHOLD = 0.4  # Below this: ask clarifying question with domain chips

INTENT_TO_DOMAIN = {
    "job": "jobs",
    "jobs": "jobs",
    "ecommerce": "products",
    "products": "products",
    "price_filter": "products",
    "food": "food",
    "travel": "travel",
    "automobiles": "automobiles",
    "real_estate": "real_estate",
}

_SHORT_FOLLOW_UP_PHRASES = (
    "which one is best",
    "best one",
    "link",
    "apply",
    "top one",
    "recommend one",
)
from nlp.intent_classifier import intent_classifier
from nlp.entity_extractor import entity_extractor
from nlp.groq_enricher import enrich_message
from nlp.recommender import append_recommendations_to_reply
from rag.rag_engine import rag_answer
from utils.helpers import normalize_text, is_knowledge_question, is_action_query
from utils.logger import logger

# Free APIs: optional Groq LLM (set GROQ_API_KEY to enable)
try:
    from apis.llm_client import (
        is_available as llm_available,
        enhance_reply as llm_enhance_reply,
        generate_reply_from_context as llm_generate_reply_from_context,
    )
except Exception:
    llm_available = lambda: False
    llm_enhance_reply = lambda *a, **k: None
    llm_generate_reply_from_context = lambda *a, **k: None

# Import all domain handlers
from domains.food_domain import FoodDomain
from domains.travel_domain import TravelDomain
from domains.products_domain import ProductsDomain
from domains.jobs_domain import JobsDomain
from domains.automobiles_domain import AutomobilesDomain
from domains.real_estate_domain import RealEstateDomain

class Chatbot:
    """
    Main chatbot orchestrator.
    
    Coordinates intent classification, entity extraction, domain routing,
    and RAG fallback to provide intelligent responses.
    """
    
    def __init__(self):
        """Initialize chatbot with all domain handlers."""
        # Initialize all domain handlers
        # Each domain handler manages its own domain-specific logic
        self.domains = {
            "travel": TravelDomain(context_manager),
            "food": FoodDomain(context_manager),
            "products": ProductsDomain(context_manager),
            "jobs": JobsDomain(context_manager),
            "automobiles": AutomobilesDomain(context_manager),
            "real_estate": RealEstateDomain(context_manager),
        }
        
        logger.logger.info("Chatbot initialized with all domain handlers")

    def _get_session_history(self, session_id: str) -> List[Dict[str, str]]:
        return conversation_sessions.get(session_id, [])

    def _record_session_turn(self, session_id: str, user_message: str, bot_response: str) -> None:
        if session_id not in conversation_sessions:
            conversation_sessions[session_id] = []
        conversation_sessions[session_id].append({"role": "user", "content": user_message})
        conversation_sessions[session_id].append({"role": "assistant", "content": bot_response})
        conversation_sessions[session_id] = conversation_sessions[session_id][-10:]

    def _apply_groq_domain_priority(
        self,
        enriched: Dict[str, Any],
        local_intent: Optional[str],
        local_confidence: float,
    ) -> Tuple[Optional[str], float]:
        """
        When Groq is confident, its domain overrides the local intent model for routing.
        Returns (routing_intent, routing_confidence) to use downstream.
        """
        routing_intent = local_intent
        routing_confidence = local_confidence
        groq_domain = enriched.get("domain")
        try:
            groq_conf = float(enriched.get("confidence", 0.0))
        except (TypeError, ValueError):
            groq_conf = 0.0

        if groq_domain and groq_domain != "unknown":
            handler_key = _GROQ_DOMAIN_TO_HANDLER.get(groq_domain)
            if handler_key:
                local_domain = INTENT_TO_DOMAIN.get(local_intent) if local_intent else None
                strongly_disagree = (
                    local_domain is not None and local_domain != handler_key
                )
                conf_threshold = (
                    GROQ_DISAGREEMENT_CONFIDENCE_THRESHOLD
                    if strongly_disagree
                    else GROQ_DOMAIN_CONFIDENCE_THRESHOLD
                )
                if groq_conf > conf_threshold:
                    context_manager.set_domain(handler_key)
                    override_intent = _GROQ_DOMAIN_TO_INTENT.get(groq_domain)
                    if override_intent is not None:
                        routing_intent = override_intent
                    routing_confidence = max(routing_confidence, groq_conf)
                    context_manager.update_metadata("domain_source", "groq")
                    context_manager.update_metadata("groq_domain", groq_domain)
                    logger.logger.info(
                        f"Groq domain priority: {groq_domain} -> {handler_key} "
                        f"(conf={groq_conf:.2f}, local_intent={local_intent}, "
                        f"disagree={strongly_disagree})"
                    )
        return routing_intent, routing_confidence

    def _merge_enrichment_into_context(self, enriched: Dict[str, Any]) -> None:
        """Apply Groq-extracted entities to context manager."""
        ents = enriched.get("entities") or {}
        if ents.get("location"):
            loc = str(ents["location"]).lower()
            context_manager.set_entity("user_location", loc)
            context_manager.set_entity("city", loc)
            context_manager.set_entity("food_city", loc)
            context_manager.set_entity("job_city", loc)
        if ents.get("budget") is not None:
            context_manager.set_entity("budget", ents["budget"])
        if ents.get("item"):
            item = str(ents["item"]).lower()
            active = context_manager.get_domain()
            if active == "food":
                context_manager.set_entity("food_item", item)
            elif active == "products":
                context_manager.set_entity("ecommerce_product", item)
            else:
                context_manager.set_entity("food_item", item)
        if ents.get("preference"):
            pref = str(ents["preference"]).lower()
            context_manager.set_entity("food_preference", pref)
            if "cheap" in pref or "sasta" in pref:
                context_manager.set_entity("price_range", "budget")
        if ents.get("destination"):
            dest = str(ents["destination"]).lower()
            context_manager.set_entity("trip_place", dest)
            context_manager.set_entity("destination", dest)
            context_manager.set_entity("flight_dest", dest)
        if ents.get("job_type"):
            context_manager.set_entity("employment_type", str(ents["job_type"]).lower())

        context_manager.update_metadata("groq_enrichment", enriched)
        context_manager.update_metadata("groq_language", enriched.get("language"))

    def _personalize_response(self, response: str, enriched: Dict[str, Any]) -> str:
        """Light personalization using extracted entities (does not change response shape)."""
        if not response:
            return response
        ents = enriched.get("entities") or {}
        hints = []
        if ents.get("location"):
            hints.append(f"📍 {ents['location'].title()}")
        if ents.get("item"):
            hints.append(f"🔎 {ents['item']}")
        if ents.get("budget"):
            hints.append(f"💰 ~PKR {ents['budget']:,}")
        if ents.get("destination"):
            hints.append(f"✈️ {ents['destination'].title()}")
        if hints and enriched.get("confidence", 0) >= 0.5:
            prefix = " ".join(hints)
            if prefix not in response:
                return f"{prefix}\n\n{response}"
        return response

    def _finalize_with_enrichment(
        self,
        response: str,
        enriched: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Append follow-up question and attach enrichment to meta."""
        meta = dict(meta or {})
        meta["groq_enrichment"] = enriched
        response = self._personalize_response(response, enriched)
        conf = float(enriched.get("confidence", 0.0))
        follow_up = enriched.get("follow_up_question")
        if conf < 0.75 and follow_up and isinstance(follow_up, str) and follow_up.strip():
            if follow_up.strip() not in response:
                response = f"{response}\n\n{follow_up.strip()}"
        return response, meta

    def process(
        self, user_input: str, session_id: str = "default"
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        """
        Process user input and generate response.

        Main processing pipeline:
        1. Normalize input text
        2. Handle special cases (greetings, exit, identity)
        3. Classify intent
        4. Extract entities
        5. Route to domain or RAG
        6. Generate response

        Args:
            user_input: User's input text

        Returns:
            Tuple of (response text, suggested reply strings, optional meta dict e.g. {"low_confidence": True})
        """
        try:
            return self._process_impl(user_input, session_id=session_id)
        except Exception as e:
            logger.log_error(e, "Chatbot.process")
            return (
                "Something went wrong processing your message. "
                "Please try again or rephrase your question.",
                [],
                {},
            )

    def _process_impl(
        self, user_input: str, session_id: str = "default"
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        """Internal implementation of process(). Returns (reply_text, suggested_replies, meta)."""
        text = normalize_text(user_input)
        logger.log_user_input(user_input)
        
        special_response = self._handle_special_cases(text)
        if special_response:
            logger.log_response(special_response)
            self._record_session_turn(session_id, user_input, special_response)
            return special_response, [], {}

        # Classify intent using ML model (unchanged — runs first)
        intent, confidence = intent_classifier.classify(text)
        context_manager.update_metadata("last_intent", intent)
        context_manager.update_metadata("last_confidence", confidence)
        band = intent_classifier.get_confidence_level(confidence)
        logger.log_confidence_band(band, confidence)

        if len(user_input.strip()) < 4 or text in (
            "ok", "okay", "umm", "hmm", "lol", "haha",
            "nice", "stop", "huh", "sksk", "byee", "byeee",
            "cool", "fine", "sure", "alright", "great",
        ):
            casual = "How can I help you?"
            logger.log_response(casual)
            self._record_session_turn(session_id, user_input, casual)
            return casual, [], {}

        # Groq enrichment layer (runs second, after intent detection)
        history = self._get_session_history(session_id)
        enriched = enrich_message(
            user_input,
            detected_intent=intent or "unknown",
            conversation_history=history,
        )
        logger.logger.info(f"GROQ enrichment: {enriched}")
        routing_intent, routing_confidence = self._apply_groq_domain_priority(
            enriched, intent, confidence
        )
        self._merge_enrichment_into_context(enriched)
        search_text = normalize_text(enriched.get("enriched_query") or user_input)

        # Extract entities (use enriched query for search-oriented extraction)
        entities = entity_extractor.extract_entities(
            search_text, domain=context_manager.get_domain()
        )
        
        # Update context with extracted entities
        # This enables multi-turn conversations
        for entity_type, value in entities.items():
            if value:
                context_manager.set_entity(entity_type, value)
        
        # Determine if RAG should be used for knowledge questions (when domain doesn't handle).
        should_use_rag = intent_classifier.should_use_rag(routing_confidence)
        if not should_use_rag and is_knowledge_question(search_text):
            if routing_confidence < 0.7:
                should_use_rag = True

        # Prefer domain routing (uses Groq intent/domain when Groq confidence > 0.6)
        response, suggestions = self._route_to_domain(
            routing_intent, search_text, routing_confidence, entities
        )

        if response:
            if llm_available() and llm_generate_reply_from_context:
                generated = llm_generate_reply_from_context(
                    user_input, "domain_reply", response
                )
                if generated:
                    response = generated
                elif llm_enhance_reply:
                    enhanced = llm_enhance_reply(response, user_input)
                    if enhanced:
                        response = enhanced
            elif llm_available() and llm_enhance_reply:
                enhanced = llm_enhance_reply(response, user_input)
                if enhanced:
                    response = enhanced
            rec_domain = context_manager.get_domain() or context_manager.get_metadata("last_domain")
            if rec_domain in ("flights", "trips"):
                rec_domain = "travel"
            response = append_recommendations_to_reply(
                response, rec_domain, entities=entities, limit=5
            )
            response, meta = self._finalize_with_enrichment(response, enriched, {})
            logger.log_response(response)
            self._record_session_turn(session_id, user_input, response)
            return response, suggestions if isinstance(suggestions, list) else [], meta

        # Try RAG only when no domain handled and confidence suggests knowledge query
        if should_use_rag:
            rag_response, rag_reason = rag_answer(search_text)
            if (
                rag_response is not None
                and isinstance(rag_response, str)
                and rag_response.strip()
            ):
                context_manager.update_metadata("last_rag_reason", rag_reason)
                logger.log_domain("RAG", "Knowledge question answered via RAG")
                if llm_available() and llm_generate_reply_from_context:
                    llm_reply = llm_generate_reply_from_context(user_input, "rag", rag_response)
                    if llm_reply and isinstance(llm_reply, str) and llm_reply.strip():
                        rag_response = llm_reply
                logger.log_response(rag_response)
                rag_out = f"Here's what I found 👇\n\n{rag_response}"
                rag_out, meta = self._finalize_with_enrichment(rag_out, enriched, {})
                self._record_session_turn(session_id, user_input, rag_out)
                return rag_out, [], meta

        if not routing_intent or routing_confidence < CONFIDENCE_THRESHOLD:
            logger.log_domain("clarification", "Low confidence — offering domain chips")
            suggestions = ["Travel", "Food", "E-Commerce", "Jobs", "Automobiles"]
            clarify = (
                "I'm not sure what you need yet.\n"
                "Is your question about travel, food, e-commerce, jobs, automobiles, or real estate?"
            )
            clarify, meta = self._finalize_with_enrichment(
                clarify, enriched, {"low_confidence": True}
            )
            self._record_session_turn(session_id, user_input, clarify)
            return clarify, suggestions, meta

        fallback = (
            "I can help with Food, Jobs, Travel, Automobiles, E-Commerce and Real Estate. "
            "How can I assist you?"
        )
        fallback, meta = self._finalize_with_enrichment(fallback, enriched, {})
        logger.log_response(fallback)
        self._record_session_turn(session_id, user_input, fallback)
        return fallback, [], meta

    def _handle_special_cases(self, text: str) -> Optional[str]:
        """
        Handle special cases that don't need domain routing.
        
        Special cases include:
        - Greetings
        - Exit commands
        - Thanks
        - Identity questions
        
        Args:
            text: Normalized user input
            
        Returns:
            Response text or None if not a special case
        """
        # Handle identity questions
        # Users often ask who the bot is
        if any(q in text for q in [
            "who are you", "what is your name", "your name",
            "who built you", "who created you"
        ]):
            return (
                "🤖 I am ScrapBot, your AI-powered assistant!\n\n"
                "I can help you with:\n"
                "✈️ Travel - trips and destinations\n"
                "🚗 Automobiles - cars and vehicles\n"
                "🛍️ E-Commerce - products and deals\n"
                "🍕 Food - restaurants and delivery\n"
                "💼 Jobs - career opportunities\n"
                "🏠 Real Estate - properties for rent and sale\n\n"
                "Just ask me anything!"
            )
        
        goodbye_words = [
            "bye", "byee", "byeee", "goodbye",
            "allah hafiz", "khuda hafiz", "allah hafiz",
            "allah", "hafiz", "khuda", "alvida",
            "take care", "see you", "later", "cya", "tata",
            "phir milenge", "chalte hain", "chalta hun",
        ]
        if any(word in text.lower() for word in goodbye_words):
            return (
                "Allah Hafiz! Have a great day!\n"
                "Hope to see you soon. Take care! 👋"
            )
        
        # Handle exit commands
        if text in ["exit", "bye", "quit", "goodbye"]:
            context_manager.reset()
            return "Allah Hafiz"
        
        # Handle thanks
        if text in [
            "thanks", "nice", "thank you", "thankyou",
            "ok", "okay", "done", "shukriya", "jazakallah", "great", "perfect",
        ]:
            context_manager.reset()
            return "You're welcome 😊"
        
        # Handle greetings
        if text in ["hello", "hi", "hey", "greetings"]:
            return "Hello! How can I help you?"

        # Start over / something else – reset context and offer categories
        if any(phrase in text for phrase in ["start over", "something else", "new search", "start again"]):
            context_manager.reset()
            return (
                "Sure! What would you like help with?\n"
                "• Travel  • Food  • E-Commerce  • Jobs  • Automobiles • Real Estate"
            )

        # Help / what can you do
        if any(phrase in text for phrase in ["help", "what can you do", "what can you help", "how does this work"]):
            return (
                "I can help you with:\n"
                "✈️ **Travel** – flights, hotels, trips & tours\n"
                "🍕 **Food** – restaurants, delivery & cuisines\n"
                "🛍️ **E-Commerce** – shop products & deals\n"
                "💼 **Jobs** – careers, internships & remote work\n"
                "🚗 **Automobiles** – cars, bikes, rentals & financing\n"
                "Say *start over* anytime to switch topic."
            )
        
        return None

    def _is_short_follow_up(self, text: str) -> bool:
        t = text.strip().lower()
        return any(phrase in t for phrase in _SHORT_FOLLOW_UP_PHRASES)
    
    def _route_to_domain(
        self,
        intent: Optional[str],
        text: str,
        confidence: float = 0.0,
        entities: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], List[str]]:
        """
        Route user input to appropriate domain handler.
        
        Routing logic:
        1. Check if current domain is active and can handle input
        2. Check each domain's can_handle() method
        3. Call domain's handle() method to generate response
        4. Update context with domain state
        
        Args:
            intent: Detected intent (may be None)
            text: Normalized user input
            confidence: Intent classification confidence (0.0 to 1.0)
            
        Returns:
            Tuple of (response text or None, suggested_replies list)
        """
        def unpack(result):
            resp = result[0]
            dom = result[1] if len(result) > 1 else None
            sugg = result[2] if len(result) > 2 else []
            return resp, dom, sugg

        if self._is_short_follow_up(text):
            last_domain_results = context_manager.get_metadata("last_domain_results")
            if last_domain_results in ("flights", "trips"):
                last_domain_results = "travel"
            if last_domain_results and last_domain_results in self.domains:
                domain = self.domains[last_domain_results]
                context_manager.set_domain(last_domain_results)
                logger.log_domain(last_domain_results, "Short follow-up via last_domain_results")
                result = domain.handle(text, entities=entities)
                response, new_domain, suggestions = unpack(result)
                if response:
                    context_manager.update_metadata("last_domain", last_domain_results)
                    context_manager.update_metadata("last_domain_results", last_domain_results)
                    context_manager.update_metadata("last_results", (response[:80] + "…") if len(response) > 80 else response)
                if new_domain:
                    context_manager.set_domain(new_domain)
                elif new_domain is None and response:
                    context_manager.reset()
                if response:
                    return response, suggestions

        # High-confidence intent routing overrides active domain hijacking
        if intent and confidence >= CONFIDENCE_HIGH:
            target_domain = INTENT_TO_DOMAIN.get(intent)
            if target_domain and target_domain in self.domains:
                active_domain_name = context_manager.get_domain()
                if active_domain_name and active_domain_name != target_domain:
                    context_manager.reset()
                context_manager.set_domain(target_domain)
                domain = self.domains[target_domain]
                logger.log_domain(target_domain, f"High-confidence intent override: {intent}")
                result = domain.handle(text, entities=entities)
                response, new_domain, suggestions = unpack(result)
                if response:
                    context_manager.update_metadata("last_domain", target_domain)
                    context_manager.update_metadata("last_domain_results", target_domain)
                    context_manager.update_metadata("last_results", (response[:80] + "…") if len(response) > 80 else response)
                if new_domain:
                    context_manager.set_domain(new_domain)
                elif new_domain is None and response:
                    context_manager.reset()
                if response:
                    return response, suggestions

        # Check active domain first
        active_domain_name = context_manager.get_domain()
        if active_domain_name and active_domain_name in self.domains:
            domain = self.domains[active_domain_name]
            if domain.can_handle(intent, text):
                result = domain.handle(text, entities=entities)
                response, new_domain, suggestions = unpack(result)
                if response:
                    context_manager.update_metadata("last_domain", active_domain_name)
                    context_manager.update_metadata("last_domain_results", active_domain_name)
                    context_manager.update_metadata("last_results", (response[:80] + "…") if len(response) > 80 else response)
                if new_domain:
                    context_manager.set_domain(new_domain)
                elif new_domain is None and response:
                    context_manager.reset()
                if response:
                    return response, suggestions

        # Short-term memory: for ambiguous follow-ups, try last_domain first
        last_domain = context_manager.get_metadata("last_domain")
        if last_domain in ("flights", "trips"):
            last_domain = "travel"
        if last_domain and last_domain in self.domains and confidence < CONFIDENCE_HIGH:
            domain = self.domains[last_domain]
            if domain.can_handle(intent, text):
                context_manager.set_domain(last_domain)
                logger.log_domain(last_domain, f"Biased by memory (ambiguous follow-up)")
                result = domain.handle(text, entities=entities)
                response, new_domain, suggestions = unpack(result)
                if response:
                    context_manager.update_metadata("last_domain", last_domain)
                    context_manager.update_metadata("last_domain_results", last_domain)
                    context_manager.update_metadata("last_results", (response[:80] + "…") if len(response) > 80 else response)
                if new_domain:
                    context_manager.set_domain(new_domain)
                elif new_domain is None:
                    context_manager.reset()
                if response:
                    return response, suggestions

        for domain_name, domain in self.domains.items():
            if domain.can_handle(intent, text):
                context_manager.set_domain(domain_name)
                conf_level = intent_classifier.get_confidence_level(confidence)
                logger.log_domain(domain_name, f"Intent: {intent}, Confidence: {conf_level}")
                result = domain.handle(text, entities=entities)
                response, new_domain, suggestions = unpack(result)
                if response:
                    context_manager.update_metadata("last_domain", domain_name)
                    context_manager.update_metadata("last_domain_results", domain_name)
                    context_manager.update_metadata("last_results", (response[:80] + "…") if len(response) > 80 else response)
                if new_domain:
                    context_manager.set_domain(new_domain)
                elif new_domain is None:
                    context_manager.reset()
                return response, suggestions

        return None, []


# Create singleton chatbot instance
# All modules use the same chatbot instance
chatbot = Chatbot()


def chatbot_response(
    user_input: str, session_id: str = "default"
) -> Tuple[str, List[str], Dict[str, Any]]:
    """
    Convenience function for chatbot processing.

    Called from app.py (GUI) to get bot response and optional suggested replies.

    Args:
        user_input: User's input text
        session_id: Conversation session key for Groq context memory

    Returns:
        Tuple of (response text, list of suggested reply strings, meta e.g. {"low_confidence": True})
    """
    best_keywords = [
        "which one is best", "which is best",
        "best one", "which one", "recommend one", "top one",
        "suggest one", "which should i", "which one should",
        "best for me", "suitable for me", "which one is suitable",
    ]

    casual_keywords = [
        "ok", "okay", "umm", "hmm", "lol",
        "haha", "nice", "stop", "huh", "sksk",
        "cool", "fine", "sure", "alright", "great", "wow", "oh",
        "okay cool", "sounds good", "buy",
    ]

    text_lower = user_input.lower().strip()

    if text_lower == "buy" and context_manager.get_domain() == "food":
        result = chatbot.domains["food"].handle(user_input)
        reply = result[0]
        suggestions = result[2] if len(result) > 2 else []
        if reply:
            context_manager.update_metadata("last_result_domain", "food")
        return reply, suggestions if isinstance(suggestions, list) else [], {}

    if text_lower in casual_keywords or len(text_lower) <= 3:
        return "😊 How can I help you?", [], {}

    if any(keyword in text_lower for keyword in best_keywords):
        last_domain = context_manager.get_metadata("last_result_domain")
        if last_domain:
            if last_domain in ("flights", "trips"):
                last_domain = "travel"
            if last_domain in chatbot.domains:
                context_manager.set_entity("asking_best", True)
                result = chatbot.domains[last_domain].handle(user_input)
                context_manager.set_entity("asking_best", None)
                reply = result[0]
                suggestions = result[2] if len(result) > 2 else []
                if reply:
                    context_manager.update_metadata("last_result_domain", last_domain)
                return reply, suggestions if isinstance(suggestions, list) else [], {}

    if context_manager.get_domain() == "food" and user_input.strip() in ("Order", "Dine in", "Dine In"):
        result = chatbot.domains["food"].handle(user_input)
        reply = result[0]
        suggestions = result[2] if len(result) > 2 else []
        if reply:
            context_manager.update_metadata("last_result_domain", "food")
        return reply, suggestions if isinstance(suggestions, list) else [], {}

    property_keywords = [
        "property", "house", "flat", "apartment",
        "ghar", "makaan", "makan", "plot", "villa", "studio",
        "rent flat", "buy house", "buy flat", "buy property",
        "buy apartment", "buy home", "ghar chahiye", "makaan chahiye",
    ]
    if any(kw in text_lower for kw in property_keywords):
        intent = "real_estate"
        confidence = 0.95
        context_manager.set_domain("real_estate")

    prev_results = context_manager.get_metadata("last_results")
    out = chatbot.process(user_input, session_id=session_id)
    reply = out[0]
    suggestions = out[1] if len(out) > 1 else []
    meta = out[2] if len(out) > 2 else {}
    if reply and context_manager.get_metadata("last_results") != prev_results:
        result_domain = context_manager.get_metadata("last_domain")
        if result_domain:
            context_manager.update_metadata("last_result_domain", result_domain)
    return reply, suggestions, meta
