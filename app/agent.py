# app/agent.py

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
from typing import Dict, List, Any, Optional, Union

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from transformers import pipeline
from app.memory_engine.utils import extract_text


# === Sub-agents and routing ===
from app.sub_agents import sub_agents, route_to_agent

# === Memory Engine Imports ===
from app.memory_engine.memory_manager import MemoryManager
from app.memory_engine.memory_summarizer import MemorySummarizer
from app.memory_engine.context_injector import ContextInjector

# === Config ===
LOCATION = "us-central1"
LLM = "gemini-2.0-flash-001"

# === Tools ===
@tool
def search(query: str) -> str:
    """Simulates a web search. Use it to get information on weather."""
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."

tools = [search]

# === LLM Setup ===
def get_llm():
    return ChatVertexAI(
        model=LLM,
        location=LOCATION,
        temperature=0.7,
        max_tokens=1024,
        streaming=True,
    ).bind_tools(tools)

# Global LLM instance
llm = get_llm()

# === Emotion Classifier (Local) ===
import os
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "goemotions"))
FALLBACK_MODEL = "SamLowe/roberta-base-go_emotions"
emotion_cache = {}

def get_emotion_classifier():
    print("Loading emotion classifier...")
    start_time = time.time()
    try:
        classifier = pipeline(
            task="text-classification",
            model=model_path,
            tokenizer=model_path,
            return_all_scores=True
        )
        _ = classifier("This is a warm-up text")
        print(f"Local emotion classifier loaded in {time.time() - start_time:.2f} seconds")
        return classifier
    except Exception as e:
        print(f"Error initializing local classifier: {e}")
        try:
            print(f"Attempting to load fallback model: {FALLBACK_MODEL}")
            fallback_start = time.time()
            classifier = pipeline(
                task="text-classification",
                model=FALLBACK_MODEL,
                return_all_scores=True
            )
            print(f"Successfully loaded fallback model in {time.time() - fallback_start:.2f} seconds")
            return classifier
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            class DummyClassifier:
                def __call__(self, text, **kwargs):
                    print("Using dummy classifier")
                    return [[
                        {"label": "neutral", "score": 0.7},
                        {"label": "joy", "score": 0.1},
                        {"label": "sadness", "score": 0.1},
                        {"label": "anger", "score": 0.1}
                    ]]
            return DummyClassifier()

emotion_classifier = get_emotion_classifier()

# def extract_text(user_input: Union[str, List, Dict]) -> str:
#     """Extract text content from various input formats."""
#     if isinstance(user_input, list) and user_input:
#         if isinstance(user_input[0], dict):
#             if 'text' in user_input[0]:
#                 return user_input[0]['text']
#             elif 'content' in user_input[0]:
#                 return user_input[0]['content']
#     if isinstance(user_input, dict):
#         if 'text' in user_input:
#             return user_input['text']
#         elif 'content' in user_input:
#             return user_input['content']
#     return str(user_input)

def analyze_emotion(user_input: Union[str, List[Dict[str, Any]]]) -> str:
    """Analyze emotion in text and return the primary emotion."""
    try:
        text = extract_text(user_input)
        if text in emotion_cache:
            cached_emotion = emotion_cache[text]
            print(f"[Agent] Emotion cache hit! Returning cached emotion: {cached_emotion}")
            return cached_emotion
        
        original_length = len(text)
        if original_length > 1000:
            text = text[:1000]
            print(f"[Agent] Truncated input from {original_length} to 1000 chars.")
        
        display_text = text if len(text) < 100 else text[:97] + "..."
        print(f"\n=== EMOTION ANALYSIS INPUT ===\n{display_text}\n===============================")
        predictions = emotion_classifier(text)
        
        if isinstance(predictions, list) and predictions:
            if isinstance(predictions[0], list) and predictions[0]:
                sorted_emotions = sorted(predictions[0], key=lambda x: x['score'], reverse=True)
                top_emotions = sorted_emotions[:5]
                emotion_details = "\n".join([f"  {e['label']}: {e['score']:.4f}" for e in top_emotions])
                print(f"[Agent] Top emotions:\n{emotion_details}")
                top_emotion = top_emotions[0]['label']
                emotion_cache[text] = top_emotion
                return top_emotion
            elif isinstance(predictions[0], dict) and 'label' in predictions[0]:
                sorted_emotions = sorted(predictions, key=lambda x: x['score'], reverse=True)
                top_emotions = sorted_emotions[:5]
                emotion_details = "\n".join([f"  {e['label']}: {e['score']:.4f}" for e in top_emotions])
                print(f"[Agent] Top emotions:\n{emotion_details}")
                top_emotion = sorted_emotions[0]['label']
                emotion_cache[text] = top_emotion
                return top_emotion
        
        print("[Agent] No valid emotion predictions found; defaulting to neutral.")
        return "neutral"
    except Exception as e:
        print(f"[Agent] Error analyzing emotion: {e}")
        import traceback
        traceback.print_exc()
        return "neutral"

def should_continue(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


def call_model(state: MessagesState, config: RunnableConfig) -> dict[str, BaseMessage]:
    # Extract user input
    last_message = state["messages"][-1]
    if hasattr(last_message, "content"):
        user_input = last_message.content
    elif isinstance(last_message, dict) and "content" in last_message:
        user_input = last_message["content"]
    else:
        user_input = str(last_message)
    
    print(f"[Agent] New user message received: {user_input}")
    
    # For demonstration, we're using a fixed user_id.
    user_id = "user_123"

    # Update rolling memory with the new message.
    MemoryManager.update_rolling_memory(user_id, user_input)
    
    # Check if we need to summarize the conversation.
    rolling_history = MemoryManager.rolling_memory.get(user_id, [])
    summary = MemorySummarizer.summarize_if_needed(rolling_history)
    if summary:
        MemoryManager.update_session_memory(user_id, summary)
        if MemorySummarizer.is_significant(summary):
            tags = MemorySummarizer.tag_emotions(summary)
            MemoryManager.append_long_term(user_id, summary, tags)
    
    # Retrieve combined memory context
    combined_context = MemoryManager.get_combined_context(user_id)
    
    # Analyze emotion and route to appropriate sub-agent.
    detected_emotion = analyze_emotion(user_input)
    print(f"[Agent] Primary detected emotion: {detected_emotion}")
    
    clean_text = extract_text(user_input)
    selected_agent_key = route_to_agent(detected_emotion)
    sub_agent = sub_agents[selected_agent_key]
    meta_intent = getattr(sub_agent, "meta_intent", "gentle and curious")
    print(f"[Agent] Routing: emotion='{detected_emotion}' → agent='{selected_agent_key}' (meta_intent='{meta_intent}')")
    sub_response = sub_agent.handle(clean_text, detected_emotion)
    
    # Build the system prompt with memory context.
    system_message = ContextInjector.build(
        meta_intent=meta_intent,
        agent_guidance=sub_response,
        session_summary=combined_context,
        long_term_facts=None,
        additional_context=None,
    )
    
    messages_with_system = [{"type": "system", "content": system_message}] + state["messages"]
    print("\n🧠 === FINAL SYSTEM PROMPT TO LLM === 🧠\n")
    print(system_message)
    print("\n🧠 ================================ 🧠\n")
    response = llm.invoke(messages_with_system, config)
    return {"messages": response}

def create_agent():
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    return workflow.compile()

agent = create_agent()

def handle_message(message):
    """Helper function to handle messages in various contexts."""
    return agent.invoke({"messages": [{"role": "user", "content": message}]})
