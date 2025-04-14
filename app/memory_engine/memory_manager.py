# memory_manager.py
import json
import os
from typing import List, Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field, asdict
from app.memory_engine.utils import extract_text

@dataclass
class MemoryEntry:
    """Represents a single long-term memory entry."""
    summary: str
    tags: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for JSON serialization."""
        data = asdict(self)
        # Convert datetime to ISO format string for JSON serialization
        data['timestamp'] = data['timestamp'].isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create a MemoryEntry from a dictionary."""
        # Convert ISO format string back to datetime
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class MemoryManager:
    """
    Manages a user's memory consisting of:
      - Rolling Memory: A rolling window of raw messages.
      - Session Memory: Summaries generated periodically during a session.
      - Long-Term Memory: Persistent entries across sessions.
    """
    # File paths for storage
    MEMORY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "memories")
    
    # For MVP purposes, we're using in-memory dictionaries for session data
    rolling_memory: Dict[str, List[str]] = {}
    session_memory: Dict[str, List[str]] = {}
    
    # Long-term memory is now persisted to disk, but we keep a cache for performance
    _long_term_memory_cache: Dict[str, List[MemoryEntry]] = {}
    
    @classmethod
    def _ensure_memory_dir(cls) -> None:
        """Ensure the memory directory exists."""
        os.makedirs(cls.MEMORY_DIR, exist_ok=True)
        print(f"[MemoryManager] Ensuring memory directory exists: {cls.MEMORY_DIR}")
    
    @classmethod
    def _get_user_memory_path(cls, user_id: str) -> str:
        """Get the file path for a user's memory file."""
        cls._ensure_memory_dir()
        return os.path.join(cls.MEMORY_DIR, f"{user_id}_memory.json")
    
    @classmethod
    def update_rolling_memory(cls, user_id: str, message) -> None:
        """Store a message in rolling memory."""
        text = extract_text(message)
        cls.rolling_memory.setdefault(user_id, []).append(text)
        print(f"[MemoryManager] Rolling memory updated. Message stored as: {text}")
    
    @classmethod
    def update_session_memory(cls, user_id: str, session_summary: str) -> None:
        """Store a session summary for the current session."""
        cls.session_memory.setdefault(user_id, []).append(session_summary)
        print(f"[MemoryManager] Session memory updated for user '{user_id}'. Session summaries count: {len(cls.session_memory[user_id])}")
    
    @classmethod
    def _load_long_term_memories(cls, user_id: str) -> List[MemoryEntry]:
        """Load long-term memories from disk."""
        if user_id in cls._long_term_memory_cache:
            return cls._long_term_memory_cache[user_id]
            
        file_path = cls._get_user_memory_path(user_id)
        if not os.path.exists(file_path):
            cls._long_term_memory_cache[user_id] = []
            return []
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                memories = [MemoryEntry.from_dict(entry) for entry in data]
                cls._long_term_memory_cache[user_id] = memories
                print(f"[MemoryManager] Loaded {len(memories)} long-term memories for user '{user_id}' from disk.")
                return memories
        except Exception as e:
            print(f"[MemoryManager] Error loading memories for user '{user_id}': {e}")
            cls._long_term_memory_cache[user_id] = []
            return []
    
    @classmethod
    def _save_long_term_memories(cls, user_id: str) -> None:
        """Save long-term memories to disk."""
        file_path = cls._get_user_memory_path(user_id)
        try:
            memories = cls._long_term_memory_cache.get(user_id, [])
            with open(file_path, 'w') as f:
                json.dump([memory.to_dict() for memory in memories], f, indent=2)
            print(f"[MemoryManager] Saved {len(memories)} long-term memories for user '{user_id}' to disk.")
        except Exception as e:
            print(f"[MemoryManager] Error saving memories for user '{user_id}': {e}")
    
    @classmethod
    def append_long_term(cls, user_id: str, summary: str, tags: Optional[List[str]] = None) -> None:
        """Append a significant summary to the long-term memory with optional tags."""
        if tags is None:
            tags = []
        
        # Load existing memories (this populates the cache if needed)
        cls._load_long_term_memories(user_id)
        
        # Create and append the new memory
        entry = MemoryEntry(summary=summary, tags=tags)
        cls._long_term_memory_cache.setdefault(user_id, []).append(entry)
        
        # Save to disk
        cls._save_long_term_memories(user_id)
        print(f"[MemoryManager] Long-term memory appended for user '{user_id}': '{summary}'. Tags: {tags}")
    
    @classmethod
    def get_long_term_memories(cls, user_id: str) -> List[MemoryEntry]:
        """Get all long-term memories for a user."""
        return cls._load_long_term_memories(user_id)
    
    @classmethod
    def get_combined_context(cls, user_id: str) -> str:
        """
        Construct a combined context string that includes:
          - The last three rolling messages.
          - All session summaries.
          - A formatted view of the long-term memories.
        """
        context_parts = []

        # Include up to the last 3 messages from rolling memory
        rolling = cls.rolling_memory.get(user_id, [])
        if rolling:
            recent = rolling[-3:]
            recent_strings = [str(m) for m in recent]
            context_parts.append("Recent Messages:\n" + "\n".join(recent_strings))
            print(f"[MemoryManager] Retrieved {len(recent)} recent messages for user '{user_id}'.")

        # Include all session summaries
        sessions = cls.session_memory.get(user_id, [])
        if sessions:
            context_parts.append("Session Summaries:\n" + "\n".join(sessions))
            print(f"[MemoryManager] Retrieved {len(sessions)} session summaries for user '{user_id}'.")

        # Include all long-term memory entries
        long_terms = cls._load_long_term_memories(user_id)
        if long_terms:
            long_term_text = "Long-Term Memories:\n" + "\n".join(
                [f"- {entry.summary} (tags: {', '.join(entry.tags)})" for entry in long_terms]
            )
            context_parts.append(long_term_text)
            print(f"[MemoryManager] Retrieved {len(long_terms)} long-term memories for user '{user_id}'.")

        combined_context = "\n\n".join(context_parts)
        print(f"[MemoryManager] Combined context built for user '{user_id}':\n{combined_context}\n")
        return combined_context