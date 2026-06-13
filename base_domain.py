"""
Base Domain Class for Scrapbot.

This module defines the BaseDomain abstract class that all domain handlers
must inherit from. This provides a consistent interface for domain handling
and enables polymorphism.

Polymorphism:
- All domains implement the same interface (can_handle, handle)
- Chatbot can treat all domains uniformly
- Easy to add new domains without changing chatbot logic

Why base classes are used:
- Code reuse: Common functionality shared across domains
- Consistency: All domains follow same pattern
- Extensibility: New domains can be added easily
- Type safety: Ensures all domains implement required methods

How new domains can be added easily:
1. Create new domain class inheriting from BaseDomain
2. Implement can_handle() method (checks if domain can handle intent)
3. Implement handle() method (processes user input)
4. Register domain in chatbot.py
5. No changes needed to chatbot core logic
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
from context.context_manager import ContextManager


class BaseDomain(ABC):
    """
    Abstract base class for all domain handlers.
    
    All domain handlers must inherit from this class and implement:
    - can_handle(): Check if domain can handle the given intent/input
    - handle(): Process user input and return response
    """
    
    def __init__(self, domain_name: str, context_manager: ContextManager):
        """
        Initialize domain handler.
        
        Args:
            domain_name: Name of the domain (e.g., "food", "flights")
            context_manager: Context manager instance for state management
        """
        self.domain_name = domain_name
        self.context_manager = context_manager
    
    @abstractmethod
    def can_handle(self, intent: Optional[str], user_input: str) -> bool:
        """
        Check if this domain can handle the given intent or user input.
        
        This method is called by the chatbot to determine which domain
        should process the user's input. Domains can check:
        - Intent classification result
        - Keywords in user input
        - Current context state
        
        Args:
            intent: Detected intent from intent classifier (may be None)
            user_input: User's input text
            
        Returns:
            True if this domain can handle the input
        """
        pass
    
    @abstractmethod
    def handle(
        self, user_input: str, entities: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Optional[str]]:
        """
        Handle user input and generate response.
        
        This method processes the user's input within this domain's context.
        It should:
        1. Use provided entities or extract from input
        2. Update context with extracted entities
        3. Generate appropriate response
        4. Return response and updated domain (or None to reset)
        
        Args:
            user_input: User's input text
            entities: Optional pre-extracted entities from chatbot (avoids duplicate extraction)
            
        Returns:
            Tuple of (response_text, domain_name)
            - response_text: Bot's response to user
            - domain_name: Domain to remain active (None to reset context)
        """
        pass
    
    def get_domain_name(self) -> str:
        """
        Get the name of this domain.
        
        Returns:
            Domain name string
        """
        return self.domain_name
    
    def is_active(self) -> bool:
        """
        Check if this domain is currently active in context.
        
        Returns:
            True if this domain is the active domain
        """
        return self.context_manager.get_domain() == self.domain_name
    
    def set_active(self):
        """
        Set this domain as active in context.
        """
        self.context_manager.set_domain(self.domain_name)
    
    def reset_context(self):
        """
        Reset domain-specific context.
        
        Called when conversation completes or user switches domains.
        """
        self.context_manager.reset_domain()
