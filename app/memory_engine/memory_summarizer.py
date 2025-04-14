# memory_summarizer.py
from typing import List, Optional, Dict, Any
import json
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers.openai_functions import PydanticAttrOutputFunctionsParser
from pydantic import BaseModel, Field

# Constants
LOCATION = "us-central1"
LLM = "gemini-2.0-flash-001"  # Using the same model as in agent.py

class MemoryEvaluation(BaseModel):
    """Schema for LLM's memory evaluation."""
    is_significant: bool = Field(
        description="Whether this conversation segment contains significant therapeutic information"
    )
    summary: str = Field(
        description="A concise summary of the key points discussed in this conversation segment"
    )
    tags: List[str] = Field(
        description="Emotional themes and topics present in this conversation"
    )
    reasoning: str = Field(
        description="Explanation of why this is or isn't significant for therapeutic memory"
    )

class MemorySummarizer:
    """
    Provides LLM-based summarization, tagging, and significance testing 
    for conversation histories.
    """
    _llm = None
    
    @classmethod
    def _get_llm(cls):
        """Lazy initialization of the LLM to avoid loading it unnecessarily."""
        if cls._llm is None:
            cls._llm = ChatVertexAI(
                model=LLM,
                location=LOCATION,
                temperature=0.2,  # Lower temperature for more reliable analysis
                max_tokens=1024,
            )
        return cls._llm
    
    @classmethod
    def summarize_if_needed(cls, chat_history: List[str], threshold: int = 10) -> Optional[str]:
        """
        If chat history exceeds the threshold, return a summary string.
        Now uses LLM for summarization.
        """
        if len(chat_history) < threshold:
            print(f"[MemorySummarizer] Chat history length ({len(chat_history)}) below threshold; no summary produced.")
            return None
        
        # We now use the evaluate_memory method which also summarizes
        last_n_messages = chat_history[-threshold:]
        conversation_text = "\n".join([f"- {msg}" for msg in last_n_messages])
        
        evaluation = cls.evaluate_memory(conversation_text)
        if evaluation:
            print(f"[MemorySummarizer] Summary produced: {evaluation['summary']}")
            return evaluation['summary']
        
        # Fallback to simple summary if LLM fails
        summary = (f"Summary: Started with '{chat_history[0]}' and most recently said '{chat_history[-1]}' "
                  f"(total {len(chat_history)} messages).")
        print(f"[MemorySummarizer] Fallback summary produced: {summary}")
        return summary
    
    # Updated section in memory_summarizer.py

    @classmethod
    def evaluate_memory(cls, conversation_text: str) -> Optional[Dict[str, Any]]:
        """
        Evaluate a conversation segment or single message to determine if it's significant,
        generate a summary, and identify relevant tags.
        
        Returns a dictionary with 'is_significant', 'summary', 'tags', and 'reasoning'.
        """
        try:
            llm = cls._get_llm()
            
            system_prompt = """
            You are an AI therapist's memory evaluation system. Your job is to:
            1. Determine if a message contains significant information worth remembering long-term
            2. Write a concise summary of the key points
            3. Identify emotional themes and topics as tags
            4. Explain your reasoning

            Focus on identifying information that would be valuable for a therapist to remember in future sessions.
            Information that might be significant includes:
            - Personal background (family, relationships, work)
            - Mental health history or challenges
            - Goals, fears, or aspirations
            - Major life events or traumas
            - Recurring themes or patterns
            - Strong emotional reactions
            - Important therapeutic breakthroughs or realizations
            
            Simple greetings, small talk, or vague statements are generally NOT significant.
            
            If the message contains almost no content (e.g., just "hi" or "hello"), mark it as not significant.

            Return your analysis in JSON format with the following fields:
            - is_significant: boolean
            - summary: string (concise summary of key points)
            - tags: list of strings (emotional themes and topics)
            - reasoning: string (explanation of significance decision)
            """
            
            human_message = f"""
            Please evaluate this message from a therapy client:

            "{conversation_text}"
            
            Remember that minor exchanges like greetings or simple acknowledgments are NOT significant for long-term memory.
            """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_message)
            ]
            
            # Get the evaluation from the LLM
            response = llm.invoke(messages)
            
            # Parse the response
            try:
                content = response.content
                # Handle the case where response might be in multiple formats
                if isinstance(content, str):
                    if '{' in content and '}' in content:
                        # Try to extract JSON from the response
                        json_str = content[content.find('{'):content.rfind('}')+1]
                        result = json.loads(json_str)
                    else:
                        print(f"[MemorySummarizer] Response doesn't contain valid JSON: {content}")
                        return None
                else:
                    print(f"[MemorySummarizer] Unexpected response format: {type(content)}")
                    return None
                
                # Validate required fields
                if not all(k in result for k in ['is_significant', 'summary', 'tags']):
                    print(f"[MemorySummarizer] Missing required fields in response: {result}")
                    return None
                
                print(f"[MemorySummarizer] Memory evaluation complete. Significant: {result['is_significant']}")
                if result['is_significant']:
                    print(f"[MemorySummarizer] Reasoning: {result.get('reasoning', 'No reasoning provided')}")
                
                return result
                
            except Exception as e:
                print(f"[MemorySummarizer] Error parsing LLM response: {e}")
                print(f"Raw response: {response.content}")
                return None
                
        except Exception as e:
            print(f"[MemorySummarizer] Error during memory evaluation: {e}")
            return None
    
    @classmethod
    def tag_emotions(cls, summary: str) -> List[str]:
        """
        Tag summary with emotional themes. Now uses evaluate_memory.
        """
        try:
            evaluation = cls.evaluate_memory(summary)
            if evaluation and 'tags' in evaluation:
                tags = evaluation['tags']
                print(f"[MemorySummarizer] Tags generated for summary: {tags}")
                return tags
        except Exception as e:
            print(f"[MemorySummarizer] Error tagging emotions: {e}")
        
        # Fallback to original implementation
        import random
        possible_tags = ["anxiety", "stress", "hope", "confusion", "joy"]
        tags = random.sample(possible_tags, k=2)
        print(f"[MemorySummarizer] Fallback tags generated: {tags}")
        return tags
    
    @classmethod
    def is_significant(cls, summary: str) -> bool:
        """
        Determines if the summary is significant enough for long-term memory storage.
        Now uses LLM evaluation.
        """
        try:
            evaluation = cls.evaluate_memory(summary)
            if evaluation and 'is_significant' in evaluation:
                significance = evaluation['is_significant']
                print(f"[MemorySummarizer] Summary significance: {significance}")
                print(f"[MemorySummarizer] Reasoning: {evaluation.get('reasoning', 'No reasoning provided')}")
                return significance
        except Exception as e:
            print(f"[MemorySummarizer] Error evaluating significance: {e}")
        
        # Fallback to original implementation
        significance = len(summary) > 50
        print(f"[MemorySummarizer] Fallback significance test: {'Significant' if significance else 'Not significant'}")
        return significance