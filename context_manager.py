"""
Context Manager for Scrapbot conversation state.

This module implements a ContextManager class that maintains conversation
state across multiple turns, enabling multi-turn conversations and
context-aware responses.

Why context management is essential:
- Multi-turn conversations: Users often provide information incrementally
  (e.g., "I want pizza" -> "in Islamabad" -> "for delivery")
- State persistence: Remembers user preferences and extracted entities
- Domain continuity: Maintains domain context until conversation completes
- Improved UX: Reduces need for users to repeat information

How context improves conversation flow:
- Without context: User must provide all info in one message
- With context: Bot remembers previous inputs and asks follow-up questions
- Example: "I want biryani" -> Bot asks city -> Bot asks delivery method

Edge cases handled:
- Domain switching: When user switches domains mid-conversation
- Context reset: After completing a transaction or user saying "thanks"
- Partial information: Handles incomplete entity extraction
"""

from typing import Optional, Dict, Any
from utils.logger import logger


class ContextManager:
    """
    Manages conversation context and state across multiple turns.
    
    This class stores:
    - Current domain (food, flights, trips, etc.)
    - Extracted entities (cities, products, dates, etc.)
    - Conversation metadata (turn count, last intent, etc.)
    
    The context is used by domain handlers to provide context-aware
    responses without requiring users to repeat information.
    """
    
    def __init__(self):
        """
        Initialize context manager with empty state.
        
        Context structure:
        - domain: Current active domain (None if no domain active)
        - entities: Dictionary of extracted entities (food_item, city, etc.)
        - metadata: Additional conversation metadata
        """
        # Current active domain
        # None indicates no active domain (general conversation)
        self.domain: Optional[str] = None
        
        # Dictionary to store extracted entities
        # Keys are entity types (food_item, city, product, etc.)
        # Values are entity values (e.g., "biryani", "islamabad")
        self.entities: Dict[str, Any] = {
            "food_item": None,
            "food_city": None,
            "food_mode": None,
            "user_location": None,  # City/area for local relevance (e.g. food, trips)
            "trip_place": None,
            "ecommerce_product": None,
            "auto_type": None,
            "auto_action": None,
            "job_title": None,
            "job_city": None,
            "flight_source": None,
            "flight_dest": None,
            "flight_date": None,
            "flight_class": None
        }
        
        # Metadata for conversation tracking and short-term memory
        # Useful for analytics, debugging, and biasing ambiguous follow-ups
        self.metadata: Dict[str, Any] = {
            "turn_count": 0,
            "last_intent": None,
            "last_confidence": None,
            "conversation_started": False,
            "last_reason": None,       # Domain grounding (e.g. food_item=biryani, city=islamabad)
            "last_rag_reason": None,  # RAG grounding (chunks/query) for explain-why
            "last_domain": None,       # Last domain that produced a result (for ambiguous follow-ups)
            "last_results": None,     # Summary of last result (e.g. count or type) for continuity
        }
    
    def set_domain(self, domain: str) -> None:
        """
        Set the active domain for current conversation.
        
        When domain is set, subsequent queries are routed to that domain
        handler. This enables multi-turn conversations within a domain.
        
        Args:
            domain: Domain name (e.g., "food", "flights", "trips")
        """
        # Log domain changes for debugging
        if self.domain != domain:
            logger.log_domain(domain, f"Switched from {self.domain}")
        
        self.domain = domain
        self.metadata["turn_count"] += 1
    
    def get_domain(self) -> Optional[str]:
        """
        Get the current active domain.
        
        Returns:
            Current domain name or None if no domain is active
        """
        return self.domain
    
    def set_entity(self, entity_type: str, value: Any) -> None:
        """
        Store an extracted entity in context.
        
        Entities are pieces of information extracted from user input
        (e.g., city names, product names, dates). Storing them in context
        allows domain handlers to use this information in subsequent turns.
        
        Args:
            entity_type: Type of entity (e.g., "food_item", "city")
            value: Entity value (e.g., "biryani", "islamabad")
        """
        if entity_type in self.entities:
            self.entities[entity_type] = value
            logger.log_entities({entity_type: value})
        else:
            # Allow dynamic entity types for extensibility
            # New domains can add their own entity types
            self.entities[entity_type] = value
    
    def get_entity(self, entity_type: str, default: Any = None) -> Any:
        """
        Retrieve an entity value from context.
        
        Args:
            entity_type: Type of entity to retrieve
            default: Default value if entity doesn't exist
            
        Returns:
            Entity value or default
        """
        return self.entities.get(entity_type, default)
    
    def has_entity(self, entity_type: str) -> bool:
        """
        Check if an entity exists in context.
        
        Args:
            entity_type: Type of entity to check
            
        Returns:
            True if entity exists and is not None
        """
        return self.entities.get(entity_type) is not None
    
    def update_metadata(self, key: str, value: Any) -> None:
        """
        Update conversation metadata.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get conversation metadata.

        Args:
            key: Metadata key
            default: Default value if key doesn't exist

        Returns:
            Metadata value or default
        """
        return self.metadata.get(key, default)

    def get_last_grounding_reason(self) -> Any:
        """
        Hook for explain-why: returns last domain reason or RAG reason.
        Domain sets last_reason (e.g. "food_item=biryani, city=islamabad");
        RAG sets last_rag_reason (e.g. {"chunks": 3, "query": "..."}).
        """
        return self.metadata.get("last_reason") or self.metadata.get("last_rag_reason")
    
    def reset(self) -> None:
        """
        Reset context to initial state.
        
        Called when:
        - User completes a transaction (says "thanks")
        - User explicitly wants to start over
        - Conversation times out (if timeout implemented)
        
        This ensures clean state for next conversation.
        """
        # Reset domain to None (no active domain)
        old_domain = self.domain
        self.domain = None
        
        # Clear all entities
        for key in self.entities:
            self.entities[key] = None
        
        # Reset metadata (but keep turn count for analytics)
        self.metadata["last_intent"] = None
        self.metadata["last_confidence"] = None
        self.metadata["last_reason"] = None
        self.metadata["last_rag_reason"] = None
        self.metadata["last_domain"] = None
        self.metadata["last_results"] = None
        
        logger.log_domain("None", f"Context reset (was: {old_domain})")
    
    def reset_domain(self) -> None:
        """
        Reset only domain-specific entities while keeping domain.
        
        Useful when switching between different queries in same domain
        without losing the domain context entirely.
        """
        # Reset domain-specific entities based on current domain
        # This allows starting a new query within same domain
        domain_entities = {
            "food": ["food_item", "food_city", "food_mode"],
            "trips": ["trip_place"],
            "ecommerce": ["ecommerce_product"],
            "products": ["ecommerce_product", "product_category"],
            "automobile": ["auto_type", "auto_action"],
            "automobiles": ["auto_type", "auto_action"],
            "jobs": ["job_title", "job_city"],
            "flights": ["flight_source", "flight_dest", "flight_date", "flight_class"]
        }
        
        if self.domain in domain_entities:
            for entity_type in domain_entities[self.domain]:
                self.entities[entity_type] = None
    
    def get_all_entities(self) -> Dict[str, Any]:
        """
        Get all entities in context.
        
        Returns:
            Dictionary of all entities
        """
        # Return only non-None entities for cleaner output
        return {k: v for k, v in self.entities.items() if v is not None}
    
    def is_domain_active(self) -> bool:
        """
        Check if a domain is currently active.
        
        Returns:
            True if domain is set and not None
        """
        return self.domain is not None
    
    def switch_domain(self, new_domain: str) -> None:
        """
        Switch to a new domain, resetting previous domain's entities.
        
        This handles the edge case of domain switching mid-conversation.
        When user switches domains, we reset old domain entities to avoid
        confusion but keep general context if needed.
        
        Args:
            new_domain: New domain to switch to
        """
        # Reset previous domain's entities
        if self.domain:
            self.reset_domain()
        
        # Set new domain
        self.set_domain(new_domain)
        
        logger.log_domain(new_domain, f"Domain switched from {self.domain}")


# Create singleton context manager instance
# All modules use the same context instance for consistency
context_manager = ContextManager()
