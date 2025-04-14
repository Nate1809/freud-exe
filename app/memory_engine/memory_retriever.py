# memory_retriever.py
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.memory_engine.memory_manager import MemoryEntry, MemoryManager

# Constants
LOCATION = "us-central1"
LLM = "gemini-2.0-flash-001"

class MemoryRetriever:
    """
    Retrieves relevant memories based on the current conversation context.
    Uses an LLM to determine which memories are most relevant.
    """
    _llm = None
    
    @classmethod
    def _get_llm(cls):
        """Lazy initialization of the LLM."""
        if cls._llm is None:
            cls._llm = ChatVertexAI(
                model=LLM,
                location=LOCATION,
                temperature=0.2,
                max_tokens=1024,
            )
        return cls._llm
    
    @classmethod
    def should_update_reasoning(cls, reasoning_history: List[Dict[str, str]]) -> bool:
        """Determine if reasoning should be updated based on age."""
        if not reasoning_history:
            return True
            
        try:
            last_timestamp = reasoning_history[-1]["timestamp"]
            last_time = datetime.fromisoformat(last_timestamp)
            current_time = datetime.now(timezone.utc)
            
            # If timezone is naive, make it aware to avoid comparison errors
            if last_time.tzinfo is None:
                last_time = last_time.replace(tzinfo=timezone.utc)
                
            return current_time - last_time > timedelta(days=30)
        except (KeyError, ValueError) as e:
            print(f"[MemoryRetriever] Error checking reasoning timestamp: {e}")
            return True
    
    @classmethod
    def generate_reasoning(cls, content: str) -> str:
        """Generate reasoning for why a memory might be important."""
        try:
            llm = cls._get_llm()
            
            system_prompt = """
            You are an AI therapist's memory reasoning system. 
            Explain briefly why the following memory might be important for a therapist to remember about their client.
            Focus on psychological insights, emotional patterns, or potential therapeutic relevance.
            Keep your explanation concise - just 1-2 sentences.
            """
            
            human_prompt = f"Memory: \"{content}\"\n\nWhy this memory matters:"
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = llm.invoke(messages)
            reasoning = response.content.strip()
            
            print(f"[MemoryRetriever] Generated reasoning: {reasoning}")
            return reasoning
        except Exception as e:
            print(f"[MemoryRetriever] Error generating reasoning: {e}")
            return "This memory may provide context for the client's situation or emotional state."
    
    @classmethod
    def get_relevant_memories(cls, 
                             user_id: str, 
                             current_message: str, 
                             max_memories: int = 5) -> List[MemoryEntry]:
        """
        Retrieve memories that are relevant to the current message.
        
        Args:
            user_id: The ID of the user
            current_message: The current message from the user
            max_memories: Maximum number of memories to return
            
        Returns:
            A list of relevant MemoryEntry objects
        """
        # Get all memories for this user
        all_memories = MemoryManager.get_long_term_memories(user_id)
        
        if not all_memories:
            print(f"[MemoryRetriever] No memories found for user '{user_id}'")
            return []
            
        if len(all_memories) <= max_memories:
            print(f"[MemoryRetriever] Returning all {len(all_memories)} memories as total count <= max_memories")
            # Check if any memories need reasoning updates
            updated_memories = cls._check_and_update_reasoning(user_id, all_memories)
            return updated_memories
            
        try:
            # Prepare memories for relevance ranking
            memory_texts = [
                f"Memory {i+1}: {memory.summary} (Tags: {', '.join(memory.tags)})"
                for i, memory in enumerate(all_memories)
            ]
            memory_text = "\n\n".join(memory_texts)
            
            system_prompt = """
            You are an AI therapist's memory retrieval system. Your job is to identify which memories 
            are most relevant to the current conversation.
            
            Review the list of stored memories and the current message from the client.
            Return the numbers of the most relevant memories, in order of relevance.
            
            For example, if memories 3, 1, and 5 are relevant, return: "3, 1, 5"
            
            Consider:
            - Direct references to past experiences mentioned in memories
            - Emotional themes that match
            - Similar situations or problems
            - Related people or relationships
            - Connected goals or aspirations
            """
            
            human_prompt = f"""
            CURRENT CLIENT MESSAGE:
            {current_message}
            
            STORED MEMORIES:
            {memory_text}
            
            Which memory numbers (if any) are most relevant to the current message? 
            Return ONLY the memory numbers in order of relevance, separated by commas.
            If none are relevant, return "none".
            """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            # Get relevance ranking from LLM
            llm = cls._get_llm()
            response = llm.invoke(messages)
            
            # Parse the response
            indices = []
            content = response.content.strip()
            
            # Handle "none" response
            if content.lower() == "none":
                print(f"[MemoryRetriever] LLM determined no memories are relevant")
                return []
                
            # Try to parse memory indices
            for part in content.split(','):
                try:
                    # Extract numbers from the response
                    clean_part = ''.join(c for c in part if c.isdigit())
                    if clean_part:
                        idx = int(clean_part) - 1  # Convert to 0-based index
                        if 0 <= idx < len(all_memories):
                            indices.append(idx)
                except ValueError:
                    continue
            
            # Get unique indices (in case of duplicates)
            unique_indices = []
            for idx in indices:
                if idx not in unique_indices and idx < len(all_memories):
                    unique_indices.append(idx)
            
            # Limit to max_memories
            relevant_indices = unique_indices[:max_memories]
            relevant_memories = [all_memories[i] for i in relevant_indices]
            
            print(f"[MemoryRetriever] Found {len(relevant_memories)} relevant memories out of {len(all_memories)}")
            for i, memory in enumerate(relevant_memories):
                print(f"[MemoryRetriever] Relevant memory {i+1}: {memory.summary[:50]}...")
                
            # Check if any memories need reasoning updates
            updated_memories = cls._check_and_update_reasoning(user_id, relevant_memories, relevant_indices)
            return updated_memories
            
        except Exception as e:
            print(f"[MemoryRetriever] Error retrieving relevant memories: {e}")
            # Fallback to most recent memories
            print(f"[MemoryRetriever] Falling back to most recent {max_memories} memories")
            recent_memories = all_memories[-max_memories:]
            # Check if any recent memories need reasoning updates
            updated_memories = cls._check_and_update_reasoning(user_id, recent_memories)
            return updated_memories
    
    @classmethod
    def _check_and_update_reasoning(cls, user_id: str, memories: List[MemoryEntry], indices: Optional[List[int]] = None) -> List[MemoryEntry]:
        """
        Check if any memories need reasoning updates and update them if needed.
        
        Args:
            user_id: The ID of the user
            memories: List of memories to check
            indices: Original indices of the memories in the full memory list (optional)
            
        Returns:
            Updated list of memories
        """
        # If no indices provided, assume sequential order
        if indices is None:
            all_memories = MemoryManager.get_long_term_memories(user_id)
            indices = []
            for memory in memories:
                # Find the index of this memory in the full list
                # This is not efficient but works for the MVP
                for i, m in enumerate(all_memories):
                    if m.summary == memory.summary and m.timestamp == memory.timestamp:
                        indices.append(i)
                        break
                else:
                    # If not found, append a placeholder
                    indices.append(-1)
        
        # Check and update reasoning for each memory
        for i, (memory, idx) in enumerate(zip(memories, indices)):
            if cls.should_update_reasoning(memory.reasoning_history):
                print(f"[MemoryRetriever] Updating reasoning for memory: {memory.summary[:30]}...")
                new_reasoning = cls.generate_reasoning(memory.summary)
                
                # Add new reasoning to history
                if not hasattr(memory, 'reasoning_history') or memory.reasoning_history is None:
                    memory.reasoning_history = []
                    
                memory.reasoning_history.append({
                    "text": new_reasoning,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
                # Save the updated memory if we have a valid index
                if idx >= 0:
                    MemoryManager.save_updated_memory(user_id, idx, memory)
        
        return memories
    
    @classmethod
    def format_relevant_memories(cls, memories: List[MemoryEntry]) -> str:
        """Format relevant memories into a string for context."""
        if not memories:
            return ""
            
        memory_lines = []
        for i, memory in enumerate(memories):
            # Format timestamp
            if isinstance(memory.timestamp, str):
                timestamp = memory.timestamp
            else:
                timestamp = memory.timestamp.strftime("%Y-%m-%d")
                
            # Get the most recent reasoning
            reasoning = "No insight available"
            if hasattr(memory, 'reasoning_history') and memory.reasoning_history:
                reasoning = memory.reasoning_history[-1]["text"]
            elif hasattr(memory, 'reasoning') and memory.reasoning:
                reasoning = memory.reasoning
                
            memory_line = f"{i+1}. [{timestamp}] {memory.summary} (Tags: {', '.join(memory.tags)})"
            memory_line += f"\n   ↪ Why it matters: {reasoning}"
                
            memory_lines.append(memory_line)
            
        return "Relevant Past Memories:\n" + "\n".join(memory_lines)