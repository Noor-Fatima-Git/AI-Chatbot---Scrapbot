"""
Structured logging system for Scrapbot.

This module provides centralized logging functionality to track:
- User inputs and interactions
- Detected intents and confidence scores
- Domain routing decisions
- Entity extraction results
- Errors and exceptions
- RAG retrieval operations

Why logging is essential:
- Debugging: Helps identify issues in production without user interaction
- Analytics: Tracks user behavior patterns and system performance
- Auditing: Maintains a record of all interactions for compliance
- Monitoring: Enables real-time system health checks

Difference between debug and production logs:
- DEBUG: Verbose information for development (all details, stack traces)
- INFO: General flow information (user inputs, intents, responses)
- WARNING: Non-critical issues (low confidence, fallbacks)
- ERROR: Critical failures (exceptions, model errors)
"""

import logging
import os
from datetime import datetime
from pathlib import Path


class ScrapbotLogger:
    """
    Centralized logger for Scrapbot application.
    
    Provides structured logging with file and console handlers.
    Logs are written to both console (for immediate feedback) and
    log files (for persistent storage and analysis).
    """
    
    def __init__(self, log_dir="logs", log_level=logging.INFO):
        """
        Initialize the logger with file and console handlers.
        
        Args:
            log_dir: Directory to store log files
            log_level: Minimum logging level (DEBUG, INFO, WARNING, ERROR)
        """
        # Create logs directory if it doesn't exist
        # This ensures log files can be written even if directory is missing
        self.log_dir = Path(log_dir)
        try:
            self.log_dir.mkdir(exist_ok=True)
        except Exception as e:
            # If we can't create logs directory, continue without file logging
            # This prevents the application from crashing on import
            print(f"Warning: Could not create logs directory: {e}")
            self.log_dir = None
        
        # Create logger instance
        # Using module name as logger name for better organization
        self.logger = logging.getLogger("scrapbot")
        self.logger.setLevel(log_level)
        
        # Prevent duplicate handlers if logger already exists
        # This avoids log duplication when module is imported multiple times
        if self.logger.handlers:
            return
        
        # Create formatters
        # Detailed format includes timestamp, level, module, and message
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console formatter is simpler for readability
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        
        # File handler - writes all logs to file
        # Using date-based filename for easier log rotation and analysis
        if self.log_dir:
            try:
                log_file = self.log_dir / f"scrapbot_{datetime.now().strftime('%Y%m%d')}.log"
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setLevel(log_level)
                file_handler.setFormatter(detailed_formatter)
                self.logger.addHandler(file_handler)
            except Exception as e:
                # If file logging fails, continue with console logging only
                print(f"Warning: Could not set up file logging: {e}")
        
        # Console handler - displays logs in terminal
        # Useful for development and debugging
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers to logger
        # File handler already added above if log_dir exists
        self.logger.addHandler(console_handler)
    
    def log_user_input(self, user_input: str):
        """
        Log user input for tracking and analysis.
        
        Args:
            user_input: The text entered by the user
        """
        self.logger.info(f"USER INPUT: {user_input}")
    
    def log_intent(self, intent: str, confidence: float):
        """
        Log detected intent and confidence score.
        
        Args:
            intent: The detected intent label
            confidence: Confidence score (0.0 to 1.0)
        """
        # Log confidence to help identify when model is uncertain
        # Low confidence may indicate need for RAG fallback
        self.logger.info(f"INTENT: {intent} (confidence: {confidence:.3f})")
        
        # Warn if confidence is low - may need fallback
        if confidence < 0.5:
            self.logger.warning(f"Low confidence intent detection: {intent} ({confidence:.3f})")
    
    def log_domain(self, domain: str, reason: str = ""):
        """
        Log domain routing decision.
        
        Args:
            domain: The selected domain name
            reason: Optional explanation for domain selection
        """
        if reason:
            self.logger.info(f"DOMAIN: {domain} - {reason}")
        else:
            self.logger.info(f"DOMAIN: {domain}")
    
    def log_entities(self, entities: dict):
        """
        Log extracted entities.
        
        Args:
            entities: Dictionary of extracted entities (e.g., {'city': 'islamabad', 'food': 'biryani'})
        """
        if entities:
            self.logger.info(f"ENTITIES EXTRACTED: {entities}")
    
    def log_rag_query(self, query: str, results_count: int = 0):
        """
        Log RAG retrieval operations.
        
        Args:
            query: The user query used for retrieval
            results_count: Number of retrieved chunks
        """
        self.logger.info(f"RAG QUERY: {query} (retrieved {results_count} chunks)")
    
    def log_error(self, error: Exception, context: str = ""):
        """
        Log errors with context.
        
        Args:
            error: The exception that occurred
            context: Additional context about where error occurred
        """
        if context:
            self.logger.error(f"ERROR in {context}: {str(error)}", exc_info=True)
        else:
            self.logger.error(f"ERROR: {str(error)}", exc_info=True)
    
    def log_confidence_band(self, band: str, confidence: float):
        """
        Log confidence band for intent (high/medium/low).

        Args:
            band: "high", "medium", or "low"
            confidence: Raw confidence score (0.0 to 1.0)
        """
        self.logger.info(f"CONFIDENCE_BAND: {band} ({confidence:.3f})")
        if band == "low":
            self.logger.warning("Low confidence — clarification or RAG may be used")

    def log_response(self, response: str):
        """
        Log bot response.

        Args:
            response: The response text sent to user
        """
        if not isinstance(response, str):
            self.logger.warning("BOT RESPONSE: [non-string response, skipped]")
            return
        # Truncate very long responses for readability
        display_response = response[:200] + "..." if len(response) > 200 else response
        self.logger.info(f"BOT RESPONSE: {display_response}")


# Create singleton logger instance
# This ensures all modules use the same logger configuration
logger = ScrapbotLogger()
