from bot_interface import VocabBotInterface
from typing import List, Dict, Optional
import asyncio
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

class RateLimitError(Exception):
    """Raised when the Realtime API responds with a rate_limit_exceeded failure."""
    pass


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
        self._input_tokens = 0
        self._output_tokens = 0

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

        # Try to access Realtime API, with retries for transient connection errors
        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
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
                return

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
                if attempt < max_attempts:
                    wait = 2 ** attempt  # 2, 4, 8 seconds
                    print(f"[WARNING] Connection attempt {attempt}/{max_attempts} failed: {e}. "
                          f"Retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    raise RuntimeError(
                        f"Failed to initialize Realtime API after {max_attempts} attempts: {e}\n"
                        f"Cannot proceed with Realtime architecture evaluation."
                    )

    async def _open_connection(self) -> None:
        """Open a fresh WebSocket connection and configure the session."""
        self.connection_manager = self.client.beta.realtime.connect(
            model="gpt-4o-realtime-preview"
        )
        self.connection = await self.connection_manager.__aenter__()
        await self.connection.session.update(
            session={
                "modalities": ["text"],
                "instructions": self.system_instructions,
                "temperature": 0.8,
            }
        )

    async def _reconnect_and_replay(self) -> None:
        """Close the current connection, open a fresh one, and replay history.

        Called when the session appears to have expired or dropped silently.
        Replays all previous conversation turns so the model has full context.
        """
        print("[INFO] Reconnecting and replaying conversation history...")

        # Close old connection
        if self.connection_manager:
            try:
                await self.connection_manager.__aexit__(None, None, None)
            except Exception:
                pass
            self.connection = None
            self.connection_manager = None

        # Open fresh connection with retries
        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            try:
                await self._open_connection()
                break
            except Exception as e:
                if attempt < max_attempts:
                    wait = 2 ** attempt
                    print(f"[WARNING] Reconnect attempt {attempt}/{max_attempts} failed: {e}. "
                          f"Retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    raise RuntimeError(
                        f"Failed to reconnect after {max_attempts} attempts: {e}"
                    )

        # Replay all conversation history except the last user message,
        # which send_message will re-send as a fresh item.
        history_to_replay = self.conversation_history[:-1]
        for msg in history_to_replay:
            content_type = "input_text" if msg['role'] == 'user' else "text"
            await self.connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": msg['role'],
                    "content": [{"type": content_type, "text": msg['content']}],
                }
            )

        print(f"[INFO] Reconnected. Replayed {len(history_to_replay)} history messages.")

    async def send_message(self, message: str) -> str:
        """Send message via Realtime API and get response.

        Uses the persistent connection established during initialization.
        If the session has expired (detected via an empty response), reconnects,
        replays history, and retries once.

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

        max_attempts = 5
        need_to_add_item = True  # Add the item on the first attempt or after reconnect

        for attempt in range(1, max_attempts + 1):
            try:
                if need_to_add_item:
                    await self.connection.conversation.item.create(
                        item={
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": message}]
                        }
                    )
                    need_to_add_item = False

                await self.connection.response.create()
                response_text = await self._collect_response(self.connection)

            except RateLimitError as e:
                wait = min(15 * attempt, 60)
                print(f"[WARNING] Rate limited (attempt {attempt}/{max_attempts}). "
                      f"Waiting {wait}s before retry...")
                await asyncio.sleep(wait)
                # The conversation item is already on the server — just retry response.create()
                continue

            except RuntimeError:
                raise
            except Exception as e:
                raise RuntimeError(
                    f"Realtime API request failed: {e}\n"
                    f"Cannot complete Realtime architecture evaluation."
                )

            if not response_text:
                if attempt < max_attempts:
                    # Unknown empty response — session may have expired; reconnect and replay
                    print(f"[WARNING] Empty response on attempt {attempt}. "
                          f"Reconnecting to recover session...")
                    await self._reconnect_and_replay()
                    need_to_add_item = True  # re-add the current user message after replay
                    continue
                raise RuntimeError(
                    "Empty response from Realtime API after reconnect. Cannot recover."
                )

            # Store in history
            self.conversation_history.append({
                'role': 'assistant',
                'content': response_text
            })
            return response_text

        raise RuntimeError(f"Realtime API failed after {max_attempts} attempts.")

    async def _collect_response(self, connection) -> str:
        """Collect streaming response from Realtime API.

        Args:
            connection: Active Realtime API connection

        Returns:
            Complete response text (empty string if response had no output)
        """
        response_parts = []
        events_seen = []

        try:
            async for event in connection:
                events_seen.append(event.type)
                if event.type == "response.text.delta":
                    response_parts.append(event.delta)
                elif event.type == "response.done":
                    resp_obj = getattr(event, 'response', None)
                    usage = getattr(resp_obj, 'usage', None)
                    if usage:
                        self._input_tokens += getattr(usage, 'input_tokens', 0)
                        self._output_tokens += getattr(usage, 'output_tokens', 0)
                    # If no text deltas arrived, inspect the response status
                    if not response_parts and resp_obj is not None:
                        status = getattr(resp_obj, 'status', 'unknown')
                        status_details = getattr(resp_obj, 'status_details', None)
                        error = getattr(status_details, 'error', None)
                        code = getattr(error, 'code', None)
                        print(f"[DEBUG] response.done — status={status!r}, "
                              f"code={code!r}, "
                              f"events_seen={events_seen}")
                        if code == 'rate_limit_exceeded':
                            raise RateLimitError(str(error))
                        # Try to extract text from the response object as a fallback
                        output = getattr(resp_obj, 'output', None) or []
                        for item in output:
                            for part in getattr(item, 'content', None) or []:
                                text = getattr(part, 'text', None)
                                if text:
                                    response_parts.append(text)
                    break
                elif event.type == "error":
                    raise RuntimeError(f"Realtime API error: {event.error}")

            return ''.join(response_parts)

        except RateLimitError:
            raise  # let send_message handle rate limit backoff
        except Exception as e:
            raise RuntimeError(f"Error collecting Realtime API response: {e}")

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history.

        Returns:
            List of user/assistant message dictionaries
        """
        return self.conversation_history.copy()

    def get_token_usage(self) -> Dict[str, int]:
        """Get cumulative token usage across all turns so far."""
        return {
            'input': self._input_tokens,
            'output': self._output_tokens,
            'total': self._input_tokens + self._output_tokens,
        }

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
