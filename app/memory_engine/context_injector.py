# context_injector.py
from typing import Optional

class ContextInjector:
    """
    Builds the system prompt combining:
      - The agent's meta intent.
      - Guidance from sub-agents.
      - Context from session and long-term memory.
      - Any additional context.
    """

    @staticmethod
    def build(meta_intent: str,
              agent_guidance: str,
              session_summary: Optional[str] = None,
              long_term_facts: Optional[str] = None,
              additional_context: Optional[str] = None) -> str:
        prompt_parts = [
            "You are Sei, a thoughtful, compassionate AI therapist.",
            f"Meta Intent: {meta_intent}",
            f"Agent Guidance: {agent_guidance}"
        ]

        if long_term_facts:
            prompt_parts.append("Long-Term Memory:")
            prompt_parts.append(long_term_facts)
            print("[ContextInjector] Added long-term memory to system prompt.")

        if session_summary:
            prompt_parts.append("Session Summary:")
            prompt_parts.append(session_summary)
            print("[ContextInjector] Added session summary to system prompt.")

        if additional_context:
            prompt_parts.append("Additional Context:")
            prompt_parts.append(additional_context)
            print("[ContextInjector] Added additional context to system prompt.")

        final_prompt = "\n\n".join(prompt_parts)
        print(f"[ContextInjector] Final system prompt built:\n{final_prompt}\n")
        return final_prompt
