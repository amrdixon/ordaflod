from abc import ABC, abstractmethod
from typing import List, Dict

class VocabBotInterface(ABC):
    """Abstract interface for vocabulary bot implementations.

    This interface allows different bot architectures (traditional LiteLLM-based
    and OpenAI Realtime API-based) to be used interchangeably in the evaluation
    framework.
    """

    @abstractmethod
    async def initialize(self, prompt: str, vocab_dict: Dict[str, str]) -> None:
        """Initialize the bot with a system prompt and vocabulary dictionary.

        Args:
            prompt: System prompt template with {{VOCABULARY_LIST}} placeholder
            vocab_dict: Dictionary mapping vocabulary words to their definitions

        The implementation should:
        1. Replace {{VOCABULARY_LIST}} placeholder with vocab_dict JSON
        2. Set up the conversation with the complete system prompt
        3. Prepare the bot to receive user messages
        """
        pass

    @abstractmethod
    async def send_message(self, message: str) -> str:
        """Send a user message and return the bot's response.

        Args:
            message: The user's message text

        Returns:
            The bot's complete response as a string

        The implementation should:
        1. Send the user message to the bot
        2. Wait for and collect the complete response
        3. Update internal conversation history
        4. Return the response text
        """
        pass

    @abstractmethod
    def get_history(self) -> List[Dict[str, str]]:
        """Get the conversation history in a standard format.

        Returns:
            List of message dictionaries with 'role' and 'content' keys
            Format: [{'role': 'user'|'assistant', 'content': '...'}, ...]

        Note: Should not include the system message, only user/assistant turns
        """
        pass

    @abstractmethod
    def is_session_complete(self) -> bool:
        """Check if the bot has indicated the session is complete.

        Returns:
            True if the bot has signaled completion (via "beep boop"), False otherwise

        The implementation should check if the last assistant message contains
        "beep boop" (case-insensitive) to determine session completion.
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up any resources held by the bot.

        This may include:
        - Closing network connections (WebSocket for Realtime API)
        - Releasing file handles
        - Cleaning up temporary data

        For stateless APIs, this may be a no-op.
        """
        pass
