# app/agent.py

import os
# Set environment variable to disable tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from transformers import pipeline
import time

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

# Initialize with return_all_scores=True to get all emotion scores
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

# === LangGraph Functions ===
def analyze_emotion(user_input):
    """Analyze emotions in text and return top emotions with scores."""
    try:
        # Check if user_input is a list containing dictionaries
        if isinstance(user_input, list) and len(user_input) > 0 and isinstance(user_input[0], dict):
            # Extract text from the first dictionary
            text = user_input[0].get('text', '')
        else:
            text = str(user_input)

        print(f"text type {type(user_input)}")
        
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
                
                # Return the top emotion label
                return top_emotions[0]['label']
            
            # Alternative format: each prediction is a dict with label/score
            elif isinstance(predictions[0], dict) and 'label' in predictions[0]:
                sorted_emotions = sorted(predictions, key=lambda x: x['score'], reverse=True)
                top_emotions = sorted_emotions[:5]
                emotion_details = "\n".join([f"  {e['label']}: {e['score']:.4f}" for e in top_emotions])
                print(f"Top emotions:\n{emotion_details}")
                return sorted_emotions[0]['label']
        
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
    
    # Get emotion and create system message
    detected_emotion = analyze_emotion(user_input)
    print(f"Primary detected emotion: {detected_emotion}")

    system_message = (
        f"You are Sei, a thoughtful, compassionate AI therapist. "
        f"Your tone is always gentle and curious. "
        f"The user's emotional tone appears to be {detected_emotion}. "
        f"Use that as a clue but rely on your own understanding too."
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