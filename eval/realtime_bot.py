from bot_interface import VocabBotInterface
from typing import List, Dict, Optional
import json
import os

# Note: This implementation uses OpenAI's Realtime API
# The Realtime API uses gpt-4o-realtime, a voice-to-voice model
try:
    from openai import AsyncOpenAI
except ImportError:
    raise ImportError(
        "OpenAI Python SDK not found. Install with: pip install openai>=1.30.0"
    )

class RealtimeBot(VocabBotInterface):
    """Adapter for OpenAI Realtime API.

    This adapter uses the OpenAI Realtime API with gpt-4o-realtime model.
    Note: The Realtime API is a voice-to-voice model, fundamentally different
    from text-based models like gpt-4o.

    If the Realtime API is not accessible via the Python SDK, this will raise
    an error rather than falling back to a different model.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the realtime bot adapter.

        Args:
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Provide via api_key parameter or "
                "OPENAI_API_KEY environment variable."
            )

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.connection_manager = None
        self.connection = None  # Persistent connection
        self.system_instructions = None
        self.conversation_history = []
        self.initialized = False

    async def initialize(self, prompt: str, vocab_dict: Dict[str, str]) -> None:
        """Initialize Realtime API session.

        Opens a persistent WebSocket connection and configures it once.

        Args:
            prompt: System prompt template with {{VOCABULARY_LIST}} placeholder
            vocab_dict: Dictionary mapping vocabulary words to their definitions

        Raises:
            RuntimeError: If Realtime API is not available
        """
        # Replace template variable with JSON vocabulary
        vocab_json = json.dumps(vocab_dict, indent=2)
        final_instructions = prompt.replace('{{VOCABULARY_LIST}}', vocab_json)
        self.system_instructions = final_instructions

        # Try to access Realtime API
        try:
            # Get connection manager
            self.connection_manager = self.client.beta.realtime.connect(
                model="gpt-4o-realtime-preview"
            )

            # Enter the context manager ONCE and keep connection alive
            self.connection = await self.connection_manager.__aenter__()

            # Configure session for text-only mode (ONCE)
            await self.connection.session.update(
                session={
                    "modalities": ["text"],  # Text-only (no audio)
                    "instructions": self.system_instructions,
                    "temperature": 0.8,
                }
            )

            print("✓ Realtime API connection established and configured")
            self.initialized = True

        except AttributeError as e:
            raise RuntimeError(
                f"Realtime API not available in OpenAI Python SDK. "
                f"The beta.realtime namespace may not exist in this SDK version. "
                f"Error: {e}\n\n"
                f"The Realtime API may require:\n"
                f"  - A newer SDK version with Realtime support\n"
                f"  - Beta access to the Realtime API\n"
                f"  - Different SDK usage pattern\n\n"
                f"Cannot proceed with Realtime architecture evaluation."
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Realtime API: {e}\n"
                f"Cannot proceed with Realtime architecture evaluation."
            )

    async def send_message(self, message: str) -> str:
        """Send message via Realtime API and get response.

        Uses the persistent connection established during initialization.

        Args:
            message: The user's message text

        Returns:
            The bot's complete response

        Raises:
            RuntimeError: If bot not initialized or Realtime API fails
        """
        if not self.initialized or not self.connection:
            raise RuntimeError("Bot not initialized. Call initialize() first.")

        # Add user message to history for tracking
        self.conversation_history.append({
            'role': 'user',
            'content': message
        })

        try:
            # Send user message via persistent connection
            await self.connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": message}]
                }
            )

            # Request response
            await self.connection.response.create()

            # Collect response from event stream
            response_text = await self._collect_response(self.connection)

            # Store in history
            self.conversation_history.append({
                'role': 'assistant',
                'content': response_text
            })

            return response_text

        except Exception as e:
            raise RuntimeError(
                f"Realtime API request failed: {e}\n"
                f"Cannot complete Realtime architecture evaluation."
            )

    async def _collect_response(self, connection) -> str:
        """Collect streaming response from Realtime API.

        Args:
            connection: Active Realtime API connection

        Returns:
            Complete response text
        """
        response_parts = []

        try:
            async for event in connection:
                if event.type == "response.text.delta":
                    response_parts.append(event.delta)
                elif event.type == "response.done":
                    break
                elif event.type == "error":
                    raise RuntimeError(f"Realtime API error: {event.error}")

            return ''.join(response_parts)

        except Exception as e:
            raise RuntimeError(f"Error collecting Realtime API response: {e}")

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history.

        Returns:
            List of user/assistant message dictionaries
        """
        return self.conversation_history.copy()

    def is_session_complete(self) -> bool:
        """Check if bot has signaled session completion via 'beep boop'.

        Returns:
            True if last assistant message contains "beep boop", False otherwise
        """
        if not self.conversation_history:
            return False

        last_msg = self.conversation_history[-1]
        return (
            last_msg['role'] == 'assistant' and
            'beep boop' in last_msg['content'].lower()
        )

    async def cleanup(self) -> None:
        """Clean up resources.

        Closes the persistent WebSocket connection.
        """
        if self.connection and self.connection_manager:
            try:
                # Exit the context manager to close the connection
                await self.connection_manager.__aexit__(None, None, None)
                print("✓ Realtime API connection closed")
            except Exception as e:
                print(f"Warning: Error closing connection: {e}")
            finally:
                self.connection = None
                self.connection_manager = None
