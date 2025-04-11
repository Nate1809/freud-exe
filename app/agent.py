# app/agent.py

import os
# Set environment variable to disable tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import functools
from typing import Dict, List, Any, Optional, Union

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from transformers import pipeline

# === Sub-agents and routing ===
from app.sub_agents import sub_agents, route_to_agent

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
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "goemotions"))
FALLBACK_MODEL = "SamLowe/roberta-base-go_emotions"

# Cache for emotion analysis results to avoid reprocessing identical inputs
emotion_cache = {}

def get_emotion_classifier():
    print("Loading emotion classifier...")
    start_time = time.time()
    
    try:
        # Try loading local model first with return_all_scores=True
        classifier = pipeline(
            task="text-classification",
            model=model_path,
            tokenizer=model_path,
            return_all_scores=True  # Get scores for all emotions
        )
        
        # Warm up with a simple example
        _ = classifier("This is a warm-up text")
        print(f"Local emotion classifier loaded in {time.time() - start_time:.2f} seconds")
        return classifier
    except Exception as e:
        print(f"Error initializing local classifier: {e}")
        
        # Fallback to specified HuggingFace model
        try:
            print(f"Attempting to load fallback model: {FALLBACK_MODEL}")
            fallback_start = time.time()
            classifier = pipeline(
                task="text-classification",
                model=FALLBACK_MODEL,
                return_all_scores=True  # Get scores for all emotions
            )
            print(f"Successfully loaded fallback model in {time.time() - fallback_start:.2f} seconds")
            return classifier
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            
            # Very simple fallback that just returns a default emotion
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

# Global classifier instance
emotion_classifier = get_emotion_classifier()

# === Helper Function for Text Extraction ===
def extract_text(user_input: Union[str, List, Dict]) -> str:
    """Extract text content from various input formats.
    
    Args:
        user_input: Input in various possible formats
        
    Returns:
        Extracted text as string
    """
    # Handle list of dictionaries format
    if isinstance(user_input, list) and len(user_input) > 0:
        if isinstance(user_input[0], dict):
            if 'text' in user_input[0]:
                return user_input[0]['text']
            elif 'content' in user_input[0]:
                return user_input[0]['content']
    
    # Handle dictionary format
    if isinstance(user_input, dict):
        if 'text' in user_input:
            return user_input['text']
        elif 'content' in user_input:
            return user_input['content']
    
    # Default case: convert to string
    return str(user_input)

# === LangGraph Functions ===
def analyze_emotion(user_input: Union[str, List[Dict[str, Any]]]) -> str:
    """Analyze emotions in text and return top emotions with scores.
    
    Args:
        user_input: Either a string or a list containing dictionaries with text
        
    Returns:
        The top emotion label as a string
    """
    try:
        # Extract text using the shared helper function
        text = extract_text(user_input)
        
        # Check cache for this exact text (can save significant processing time)
        if text in emotion_cache:
            cached_emotion = emotion_cache[text]
            print(f"Cache hit! Returning cached emotion: {cached_emotion}")
            return cached_emotion
        
        # Truncate very long inputs to avoid tokenizer issues
        original_length = len(text)
        if original_length > 1000:
            text = text[:1000]
            print(f"Truncated input from {original_length} to 1000 chars")
        
        # Print the input text (truncated for display if very long)
        display_text = text if len(text) < 100 else text[:97] + "..."
        print(f"\n=== EMOTION ANALYSIS INPUT ===\n{display_text}\n===============================")
        
        # Get prediction with all scores
        predictions = emotion_classifier(text)
        
        # Extract and sort emotions by score
        if isinstance(predictions, list) and len(predictions) > 0:
            # Most common format: list with one item containing all emotion scores
            if isinstance(predictions[0], list) and len(predictions[0]) > 0:
                # Sort by score (highest first)
                sorted_emotions = sorted(predictions[0], key=lambda x: x['score'], reverse=True)
                
                # Get top 5 emotions with scores
                top_emotions = sorted_emotions[:5]
                
                # Display top emotions with scores
                emotion_details = "\n".join([f"  {e['label']}: {e['score']:.4f}" for e in top_emotions])
                print(f"Top emotions:\n{emotion_details}")
                
                # Cache and return the top emotion label
                top_emotion = top_emotions[0]['label']
                emotion_cache[text] = top_emotion
                return top_emotion
            
            # Alternative format: each prediction is a dict with label/score
            elif isinstance(predictions[0], dict) and 'label' in predictions[0]:
                sorted_emotions = sorted(predictions, key=lambda x: x['score'], reverse=True)
                top_emotions = sorted_emotions[:5]
                emotion_details = "\n".join([f"  {e['label']}: {e['score']:.4f}" for e in top_emotions])
                print(f"Top emotions:\n{emotion_details}")
                
                top_emotion = sorted_emotions[0]['label']
                emotion_cache[text] = top_emotion
                return top_emotion
        
        print("No valid emotion predictions found, defaulting to neutral")
        return "neutral"
    except Exception as e:
        print(f"Error analyzing emotion: {e}")
        import traceback
        traceback.print_exc()
        return "neutral"

def should_continue(state: MessagesState) -> str:
    """Determine if we should continue to tools or end."""
    last_message = state["messages"][-1]
    return "tools" if last_message.tool_calls else END

def call_model(state: MessagesState, config: RunnableConfig) -> dict[str, BaseMessage]:
    """Process the message and get response from the LLM."""
    # Extract the content safely
    last_message = state["messages"][-1]
    if hasattr(last_message, "content"):
        user_input = last_message.content
    elif isinstance(last_message, dict) and "content" in last_message:
        user_input = last_message["content"]
    else:
        user_input = str(last_message)
    
    # Get emotion and route to appropriate sub-agent
    detected_emotion = analyze_emotion(user_input)
    print(f"Primary detected emotion: {detected_emotion}")
    
    # Extract clean text version for sub-agent
    clean_text = extract_text(user_input)
    
    # Route to appropriate sub-agent based on emotion
    selected_agent_key = route_to_agent(detected_emotion)
    sub_agent = sub_agents[selected_agent_key]
    
    # 🌶️ SPICE 1: Get meta_intent from sub-agent if available
    meta_intent = getattr(sub_agent, "meta_intent", "gentle and curious")
    
    # 🌶️ SPICE 2: Add smart logging of emotion → agent → intent
    print(f"Routing: emotion='{detected_emotion}' → agent='{selected_agent_key}' (meta_intent='{meta_intent}')")
    
    # Get response from sub-agent
    sub_response = sub_agent.handle(clean_text, detected_emotion)

    # 🌶️ SPICE 1: Inject meta_intent into Sei's system prompt
    system_message = (
        f"You are Sei, a thoughtful, compassionate AI therapist. "
        f"Your tone is always {meta_intent}. "
        f"Your internal reasoning should incorporate this perspective: '{sub_response}'"
    )

    # Create messages for the LLM
    messages_with_system = [{"type": "system", "content": system_message}] + state["messages"]
    
    # Get response from LLM
    response = llm.invoke(messages_with_system, config)
    return {"messages": response}

# === Build Graph ===
def create_agent():
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    return workflow.compile()

# Create the agent
agent = create_agent()

# For streamlit or other async contexts
def handle_message(message):
    """Helper function to handle messages in various contexts."""
    return agent.invoke({"messages": [{"role": "user", "content": message}]})