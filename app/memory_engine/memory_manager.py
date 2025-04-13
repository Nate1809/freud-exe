# memory_manager.py
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass, field
from app.memory_engine.utils import extract_text


@dataclass
class MemoryEntry:
    """Represents a single long-term memory entry."""
    summary: str
    tags: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

class MemoryManager:
    """
    Manages a user's memory consisting of:
      - Rolling Memory: A rolling window of raw messages.
      - Session Memory: Summaries generated periodically during a session.
      - Long-Term Memory: Persistent entries across sessions.
    """
    # For MVP purposes, we’re using in-memory dictionaries.
    rolling_memory: Dict[str, List[str]] = {}
    session_memory: Dict[str, List[str]] = {}
    long_term_memory: Dict[str, List[MemoryEntry]] = {}

    @classmethod
    def update_rolling_memory(cls, user_id: str, message) -> None:
        text = extract_text(message)
        cls.rolling_memory.setdefault(user_id, []).append(text)
        print(f"[MemoryManager] Rolling memory updated. Message stored as: {text}")

    @classmethod
    def update_session_memory(cls, user_id: str, session_summary: str) -> None:
        """Store a session summary for the current session."""
        cls.session_memory.setdefault(user_id, []).append(session_summary)
        print(f"[MemoryManager] Session memory updated for user '{user_id}'. Session summaries count: {len(cls.session_memory[user_id])}")

    @classmethod
    def append_long_term(cls, user_id: str, summary: str, tags: Optional[List[str]] = None) -> None:
        """Append a significant summary to the long-term memory with optional tags."""
        if tags is None:
            tags = []
        entry = MemoryEntry(summary=summary, tags=tags)
        cls.long_term_memory.setdefault(user_id, []).append(entry)
        print(f"[MemoryManager] Long-term memory appended for user '{user_id}': '{summary}'. Tags: {tags}")

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
            recent_strings = [extract_text(m) for m in recent]
            context_parts.append("Recent Messages:\n" + "\n".join(recent_strings))
            print(f"[MemoryManager] Retrieved {len(recent)} recent messages for user '{user_id}'.")

        # Include all session summaries
        sessions = cls.session_memory.get(user_id, [])
        if sessions:
            context_parts.append("Session Summaries:\n" + "\n".join(sessions))
            print(f"[MemoryManager] Retrieved {len(sessions)} session summaries for user '{user_id}'.")

        # Include all long-term memory entries
        long_terms = cls.long_term_memory.get(user_id, [])
        if long_terms:
            long_term_text = "Long-Term Memories:\n" + "\n".join(
                [f"- {entry.summary} (tags: {', '.join(entry.tags)})" for entry in long_terms]
            )
            context_parts.append(long_term_text)
            print(f"[MemoryManager] Retrieved {len(long_terms)} long-term memories for user '{user_id}'.")

        combined_context = "\n\n".join(context_parts)
        print(f"[MemoryManager] Combined context built for user '{user_id}':\n{combined_context}\n")
        return combined_context
