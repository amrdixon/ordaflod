from bot_interface import VocabBotInterface
from litellm import completion
from typing import List, Dict
import json

class TraditionalBot(VocabBotInterface):
    """Adapter for traditional Claude-based bot using LiteLLM.

    This adapter wraps the LiteLLM completion API to match the VocabBotInterface,
    allowing the traditional architecture (Claude Sonnet via LiteLLM) to be used
    in the unified evaluation framework.
    """

    def __init__(self, model: str = "anthropic/claude-sonnet-4-5-20250929"):
        """Initialize the traditional bot adapter.

        Args:
            model: LiteLLM model identifier (default: Claude Sonnet 4.5)
        """
        self.model = model
        self.conversation_history = []
        self.initialized = False
        self._input_tokens = 0
        self._output_tokens = 0

    async def initialize(self, prompt: str, vocab_dict: Dict[str, str]) -> None:
        """Initialize with system prompt and vocabulary dictionary.

        Args:
            prompt: System prompt template with {{VOCABULARY_LIST}} placeholder
            vocab_dict: Dictionary mapping vocabulary words to their definitions
        """
        # Replace template variable with JSON vocabulary
        vocab_json = json.dumps(vocab_dict, indent=2)
        final_prompt = prompt.replace('{{VOCABULARY_LIST}}', vocab_json)

        # Initialize conversation with system message
        self.conversation_history = [
            {'role': 'system', 'content': final_prompt}
        ]

        self.initialized = True

    async def send_message(self, message: str) -> str:
        """Send a user message and get the bot's response.

        Args:
            message: The user's message text

        Returns:
            The bot's complete response

        Raises:
            RuntimeError: If bot not initialized
        """
        if not self.initialized:
            raise RuntimeError("Bot not initialized. Call initialize() first.")

        # Add user message to history
        self.conversation_history.append({
            'role': 'user',
            'content': message
        })

        # Get response from LiteLLM (follows pattern from bots.py:69-80)
        response = completion(
            model=self.model,
            messages=self.conversation_history
        )

        # Accumulate token usage
        usage = response.get('usage', {})
        self._input_tokens += usage.get('prompt_tokens', 0)
        self._output_tokens += usage.get('completion_tokens', 0)

        # Extract assistant message
        assistant_message = response['choices'][0]['message']['content']

        # Add assistant message to history
        self.conversation_history.append({
            'role': 'assistant',
            'content': assistant_message
        })

        return assistant_message

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history excluding system message.

        Returns:
            List of user/assistant message dictionaries
        """
        # Return all messages except the system message
        return [msg for msg in self.conversation_history if msg['role'] != 'system']

    def is_session_complete(self) -> bool:
        """Check if bot has signaled session completion via 'beep boop'.

        Returns:
            True if last assistant message contains "beep boop", False otherwise
        """
        if not self.conversation_history:
            return False

        # Get last message
        last_msg = self.conversation_history[-1]

        # Check if it's an assistant message containing "beep boop"
        return (
            last_msg['role'] == 'assistant' and
            'beep boop' in last_msg['content'].lower()
        )

    def get_token_usage(self) -> Dict[str, int]:
        """Get cumulative token usage across all turns so far."""
        return {
            'input': self._input_tokens,
            'output': self._output_tokens,
            'total': self._input_tokens + self._output_tokens,
        }

    async def cleanup(self) -> None:
        """Clean up resources (no-op for stateless API)."""
        # LiteLLM completion API is stateless, no cleanup needed
        pass
