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

# Initialize with simpler parameters (no device_map)
def get_emotion_classifier():
    print("Loading emotion classifier...")
    start_time = time.time()
    
    try:
        # Simpler initialization without device_map
        classifier = pipeline(
            task="text-classification",
            model=model_path,
            tokenizer=model_path,
            top_k=1
        )
        
        # Warm up with a simple example
        _ = classifier("This is a warm-up text")
        print(f"Emotion classifier loaded in {time.time() - start_time:.2f} seconds")
        return classifier
    except Exception as e:
        print(f"Error initializing classifier: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to direct HuggingFace model
        try:
            print("Attempting to load from HuggingFace...")
            classifier = pipeline(
                task="text-classification",
                model="SamLowe/roberta-base-go_emotions",
                top_k=1
            )
            print("Successfully loaded from HuggingFace")
            return classifier
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            
            # Very simple fallback that just returns a default emotion
            class DummyClassifier:
                def __call__(self, text, **kwargs):
                    print("Using dummy classifier")
                    return [{"label": "neutral", "score": 1.0}]
            
            return DummyClassifier()

# Global classifier instance
emotion_classifier = get_emotion_classifier()

# === LangGraph Functions ===
def analyze_emotion(user_input):
    """Safely analyze the emotion in text."""
    try:
        # Ensure we're passing a string
        if isinstance(user_input, dict) and 'content' in user_input:
            text = user_input['content']
        elif hasattr(user_input, 'content'):
            text = user_input.content
        else:
            text = str(user_input)
        
        # Truncate very long inputs to avoid tokenizer issues
        text = text[:1000] if len(text) > 1000 else text
        
        # Get prediction
        prediction = emotion_classifier(text)
        
        # Extract the emotion label
        if isinstance(prediction, list) and len(prediction) > 0:
            if isinstance(prediction[0], dict) and 'label' in prediction[0]:
                return prediction[0]['label']
            elif isinstance(prediction[0], list) and len(prediction[0]) > 0:
                return prediction[0][0]['label']
        
        return "neutral"
    except Exception as e:
        print(f"Error analyzing emotion: {e}")
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
    print(f"Detected emotion: {detected_emotion}")

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