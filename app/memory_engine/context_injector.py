# context_injector.py
from typing import Optional, List

from app.memory_engine.memory_manager import MemoryEntry
from app.memory_engine.memory_retriever import MemoryRetriever

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
              user_id: Optional[str] = None,
              current_message: Optional[str] = None,
              session_summary: Optional[str] = None,
              long_term_facts: Optional[str] = None,
              relevant_memories: Optional[List[MemoryEntry]] = None,
              additional_context: Optional[str] = None) -> str:
        """
        Build the system prompt with all available context.
        
        Args:
            meta_intent: The overall intent of the agent
            agent_guidance: Guidance from sub-agents
            user_id: User ID for retrieving relevant memories
            current_message: Current message for memory relevance
            session_summary: Summary of the current session
            long_term_facts: General long-term memory facts
            relevant_memories: Pre-retrieved relevant memories
            additional_context: Any additional context
            
        Returns:
            The complete system prompt
        """
        prompt_parts = [
            "You are Sei, a thoughtful, compassionate AI therapist.",
            f"Meta Intent: {meta_intent}",
            f"Agent Guidance: {agent_guidance}"
        ]

        # Add relevant memories if provided or retrieve them
        if user_id and current_message and not relevant_memories:
            try:
                relevant_memories = MemoryRetriever.get_relevant_memories(
                    user_id=user_id,
                    current_message=current_message
                )
                if relevant_memories:
                    memory_text = MemoryRetriever.format_relevant_memories(relevant_memories)
                    prompt_parts.append(memory_text)
                    print("[ContextInjector] Added relevant memories to system prompt.")
            except Exception as e:
                print(f"[ContextInjector] Error retrieving relevant memories: {e}")
        elif relevant_memories:
            memory_text = MemoryRetriever.format_relevant_memories(relevant_memories)
            prompt_parts.append(memory_text)
            print("[ContextInjector] Added provided relevant memories to system prompt.")

        # Add long-term memory facts if available
        if long_term_facts:
            prompt_parts.append("Long-Term Memory:")
            prompt_parts.append(long_term_facts)
            print("[ContextInjector] Added long-term memory to system prompt.")

        # Add session summary if available
        if session_summary:
            prompt_parts.append("Session Summary:")
            prompt_parts.append(session_summary)
            print("[ContextInjector] Added session summary to system prompt.")

        # Add additional context if available
        if additional_context:
            prompt_parts.append("Additional Context:")
            prompt_parts.append(additional_context)
            print("[ContextInjector] Added additional context to system prompt.")

        final_prompt = "\n\n".join(prompt_parts)
        print(f"[ContextInjector] Final system prompt built:\n{final_prompt}\n")
        return final_prompt