# memory_retriever.py
from typing import List, Dict, Any, Optional
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
            return all_memories
            
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
                
            return relevant_memories
            
        except Exception as e:
            print(f"[MemoryRetriever] Error retrieving relevant memories: {e}")
            # Fallback to most recent memories
            print(f"[MemoryRetriever] Falling back to most recent {max_memories} memories")
            return all_memories[-max_memories:]
    
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
                
            memory_lines.append(f"{i+1}. [{timestamp}] {memory.summary} (Tags: {', '.join(memory.tags)})")
            
        return "Relevant Past Memories:\n" + "\n".join(memory_lines)