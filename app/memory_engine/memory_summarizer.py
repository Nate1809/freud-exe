# memory_summarizer.py
from typing import List, Optional
import random

class MemorySummarizer:
    """
    Provides summarization, tagging, and significance testing for conversation histories.
    """

    @staticmethod
    def summarize_if_needed(chat_history: List[str], threshold: int = 10) -> Optional[str]:
        """
        If chat history exceeds the threshold, return a summary string.
        For demo purposes, we generate a simple summary.
        """
        if len(chat_history) < threshold:
            print(f"[MemorySummarizer] Chat history length ({len(chat_history)}) below threshold; no summary produced.")
            return None

        # A simple summary using the first and last messages.
        summary = (f"Summary: Started with '{chat_history[0]}' and most recently said '{chat_history[-1]}' "
                   f"(total {len(chat_history)} messages).")
        print(f"[MemorySummarizer] Summary produced: {summary}")
        return summary

    @staticmethod
    def tag_emotions(summary: str) -> List[str]:
        """
        Dummy tagging: randomly assign 1-2 tags from a fixed list.
        """
        possible_tags = ["anxiety", "stress", "hope", "confusion", "joy"]
        tags = random.sample(possible_tags, k=2)
        print(f"[MemorySummarizer] Tags generated for summary: {tags}")
        return tags

    @staticmethod
    def is_significant(summary: str) -> bool:
        """
        Determines if the summary is significant enough for long-term memory storage.
        """
        significance = len(summary) > 50  # Arbitrary threshold for significance.
        print(f"[MemorySummarizer] Summary significance test: {'Significant' if significance else 'Not significant'}")
        return significance
