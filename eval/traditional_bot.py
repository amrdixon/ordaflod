from bot_interface import VocabBotInterface
from litellm import completion
from typing import List, Dict, Optional
import json
import os
from langfuse import observe, get_client as _get_langfuse_client

# Langfuse client — only active if credentials are set in .env
_langfuse_enabled = bool(os.getenv("LANGFUSE_SECRET_KEY"))
_lf = _get_langfuse_client() if _langfuse_enabled else None

class TraditionalBot(VocabBotInterface):
    """Adapter for LLM-based bot using LiteLLM.

    This adapter wraps the LiteLLM completion API to match the VocabBotInterface,
    allowing the traditional (non-realtime) architecture to be used in the unified
    evaluation framework. The model is configured via eval_unified_config.yaml.

    API keys for the chosen provider (OpenAI, Anthropic, Gemini, etc.) must be set
    in the .env file. Other providers supported by LiteLLM should work but may
    require additional setup.
    """

    def __init__(self, model: str = "anthropic/claude-sonnet-4-5-20250929"):
        """Initialize the traditional bot adapter.

        Args:
            model: LiteLLM model identifier (default: Claude Sonnet 4.5).
                   Override via eval_unified_config.yaml.
        """
        self.model = model
        self.conversation_history = []
        self.initialized = False
        self._input_tokens = 0
        self._output_tokens = 0
        self._turn_count = 0
        self._session_ctx = None
        self._session_span = None
        self._trace_id: Optional[str] = None

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

        # Open a session-level span so all turns for this eval conversation
        # are grouped as children under one trace, tagged as source=eval.
        if _langfuse_enabled:
            self._session_ctx = _lf.start_as_current_observation(
                name="eval-vocab-quiz-session",
                input={"vocab": vocab_dict},
                metadata={"word_count": len(vocab_dict), "source": "eval", "model": self.model}
            )
            self._session_span = self._session_ctx.__enter__()
            self._trace_id = _lf.get_current_trace_id()

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

        assistant_message = await self._call_llm()
        return assistant_message

    @observe(as_type='generation', capture_input=False, capture_output=False)
    async def _call_llm(self) -> str:
        """Call LiteLLM and record the generation in Langfuse."""
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

        if _langfuse_enabled:
            _lf.update_current_generation(
                name=f"turn-{self._turn_count}",
                model=self.model,
                input=self.conversation_history,
                output=assistant_message,
                usage_details={
                    "input": usage.get('prompt_tokens', 0),
                    "output": usage.get('completion_tokens', 0),
                }
            )
        self._turn_count += 1

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

    def get_trace_id(self) -> Optional[str]:
        """Return the Langfuse trace ID captured during initialize()."""
        return self._trace_id

    async def cleanup(self) -> None:
        """Clean up resources."""
        if _langfuse_enabled and self._session_ctx:
            self._session_ctx.__exit__(None, None, None)
            _lf.flush()
