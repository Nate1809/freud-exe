# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from transformers import pipeline

# Configuration
LOCATION = "us-central1"
LLM = "gemini-2.0-flash-001"

# === 1. Define tools ===
@tool
def search(query: str) -> str:
    """Simulates a web search. Use it to get information on weather."""
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."

tools = [search]

# === 2. Set up the language model ===
llm = ChatVertexAI(
    model=LLM, location=LOCATION, temperature=0.7, max_tokens=1024, streaming=True
).bind_tools(tools)

# === 3. Initialize the emotion detection pipeline ===
emotion_classifier = pipeline(
    task="text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None  # instead of return_all_scores=True
)

# === 4. Define workflow components ===
def analyze_emotion(user_input):
    """Analyzes the user's input and returns the predominant emotion."""
    try:
        print("\n--- EMOTION ANALYSIS DEBUG ---")
        print(f"Input type: {type(user_input)}")
        
        # Ensure user_input is a plain string
        if isinstance(user_input, dict) and 'content' in user_input:
            text = user_input['content']
            print(f"Extracted from dict: {text[:50]}...")
        elif hasattr(user_input, 'content'):
            text = user_input.content
            print(f"Extracted from object with content attribute: {text[:50]}...")
        else:
            text = str(user_input)
            print(f"Converted to string: {text[:50]}...")
        
        # Make sure text is a plain string with no special attributes
        text = str(text)
        
        print(f"Final text to analyze (truncated): {text[:100]}...")
        
        # Now call the emotion classifier with the plain text
        predictions = emotion_classifier(text)
        print(f"Raw predictions: {predictions}")
        
        sorted_predictions = sorted(predictions[0], key=lambda x: x['score'], reverse=True)
        print(f"Sorted predictions: {sorted_predictions}")
        
        detected_emotion = sorted_predictions[0]['label']
        print(f"Detected emotion: {detected_emotion}")
        print("--- END EMOTION ANALYSIS ---\n")
        
        return detected_emotion
    except Exception as e:
        print(f"ERROR analyzing emotion: {e}")
        import traceback
        traceback.print_exc()
        # Return a default if there's an error
        return "neutral"

def should_continue(state: MessagesState) -> str:
    """Determines whether to use tools or end the conversation."""
    last_message = state["messages"][-1]
    return "tools" if last_message.tool_calls else END

def call_model(state: MessagesState, config: RunnableConfig) -> dict[str, BaseMessage]:
    """Calls the language model and returns the response as therapist 'Sei'."""
    last_message = state["messages"][-1]
    
    # Extract text content safely
    if hasattr(last_message, "content"):
        user_input = last_message.content
    elif isinstance(last_message, dict) and "content" in last_message:
        user_input = last_message["content"]
    else:
        user_input = str(last_message)
    
    # Convert to plain string for emotion analysis
    user_input = str(user_input)
    
    detected_emotion = analyze_emotion(user_input)
    
    system_message = (
        f"You are Sei, a thoughtful, compassionate AI therapist. "
        f"You speak in a calm, supportive tone, and help users explore their thoughts "
        f"with emotional intelligence and kindness. You adapt your responses based on the user's emotional state. "
        f"The user appears to be experiencing {detected_emotion}."
    )

    # Create messages for the LLM
    messages_with_system = [{"type": "system", "content": system_message}] + state["messages"]

    response = llm.invoke(messages_with_system, config)
    return {"messages": response}

# === 5. Create the workflow graph ===
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("agent")

# === 6. Define graph edges ===
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

# === 7. Compile the workflow ===
agent = workflow.compile()