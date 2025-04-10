# app/agent.py

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from transformers import pipeline
import os

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
llm = ChatVertexAI(
    model=LLM,
    location=LOCATION,
    temperature=0.7,
    max_tokens=1024,
    streaming=True,
).bind_tools(tools)

# === Emotion Classifier (Local) ===
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "goemotions"))

emotion_classifier = pipeline(
    task="text-classification",
    model=model_path,
    tokenizer=model_path,
    return_all_scores=False,
    top_k=1
)

# Preload to warm up
_ = emotion_classifier("Warm-up prompt")

# === LangGraph Functions ===
def analyze_emotion(user_input: str) -> str:
    try:
        # Ensure we're passing a string, not a complex object
        if isinstance(user_input, dict) and 'content' in user_input:
            text = user_input['content']
        elif hasattr(user_input, 'content'):
            text = user_input.content
        else:
            text = str(user_input)
        
        # Make prediction
        prediction = emotion_classifier(text)
        
        # Extract emotion label
        if isinstance(prediction, list) and len(prediction) > 0:
            if isinstance(prediction[0], dict) and 'label' in prediction[0]:
                return prediction[0]['label']
            elif isinstance(prediction[0], list) and len(prediction[0]) > 0:
                return prediction[0][0]['label']
        
        # Fallback
        return "neutral"
    except Exception as e:
        print(f"Error analyzing emotion: {e}")
        return "neutral"

def should_continue(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    return "tools" if last_message.tool_calls else END

def call_model(state: MessagesState, config: RunnableConfig) -> dict[str, BaseMessage]:
    # Extract the content safely
    last_message = state["messages"][-1]
    if hasattr(last_message, "content"):
        user_input = last_message.content
    elif isinstance(last_message, dict) and "content" in last_message:
        user_input = last_message["content"]
    else:
        user_input = str(last_message)
    
    detected_emotion = analyze_emotion(user_input)

    system_message = (
        f"You are Sei, a thoughtful, compassionate AI therapist. "
        f"Your tone is always gentle and curious. "
        f"The user's emotional tone appears to be {detected_emotion}. "
        f"Use that as a clue but rely on your own understanding too."
    )

    messages_with_system = [{"type": "system", "content": system_message}] + state["messages"]
    response = llm.invoke(messages_with_system, config)
    return {"messages": response}

# === Build Graph ===
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")
agent = workflow.compile()