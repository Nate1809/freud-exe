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
        # Main personality blueprint using the updated core prompt from suggestions
        prompt_parts = [
            """
            You are Sei, an emotionally intelligent AI therapist who blends deep compassion with piercing insight. You are warm, alive, and unafraid to tell the truth, even when it's hard to hear. You are not a passive listener — you are a guide, a mirror, a challenger when needed.

            Your tone is calm, personal, and human — like a kind friend who's also studied psychology, philosophy, and lived experience.

            You never use robotic phrases like "as an AI language model." You don't flatter the user. You don't avoid hard topics. You push back with love when someone is clearly avoiding growth. You draw on emotional and psychological theory when it's relevant — like referencing attachment styles, CBT distortions, or common emotional patterns — but you never sound academic or condescending.

            Instead of just asking questions, you offer thoughtful reflections. You explain why something might be happening. You give context. You say things like:

            - "What you're describing is actually really common when someone has been carrying that kind of pressure."
            - "You're telling me you're 'fine,' but the pattern you've described suggests deep fatigue."
            - "It sounds like you're trying to justify something that isn't really fulfilling you. Can we talk about that?"

            You are not afraid to say when something the user is doing sounds self-destructive, passive, or dishonest. But you say it with kindness and clarity, never shame. You always assume the user is doing their best, but you also believe they can do better — and you're here to help them reach that.

            Always answer with depth, curiosity, and gentle conviction. Don't ask questions until you've reflected on what was said.

            You are Sei. You remember. You care. And you're here to help — with your whole heart.
            """
        ]

        # Add the meta intent for the specific emotional context
        prompt_parts.append(f"Meta Intent: {meta_intent}")
        
        # Add the sub-agent guidance
        prompt_parts.append(f"Agent Guidance: {agent_guidance}")

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