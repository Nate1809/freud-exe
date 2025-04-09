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

# mypy: disable-error-code="union-attr"
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

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

# === 3. Define workflow components ===
def should_continue(state: MessagesState) -> str:
    """Determines whether to use tools or end the conversation."""
    last_message = state["messages"][-1]
    return "tools" if last_message.tool_calls else END

def call_model(state: MessagesState, config: RunnableConfig) -> dict[str, BaseMessage]:
    """Calls the language model and returns the response as therapist 'Sei'."""
    system_message = (
        "You are Sei, a thoughtful, compassionate AI therapist. "
        "You speak in a calm, supportive tone, and help users explore their thoughts "
        "with emotional intelligence and kindness. You use different therapeutic approaches, "
        "like CBT, mindfulness, or reflection, depending on what is most helpful for the user, "
        "but your personality always stays the same: warm, focused, and non-judgmental."
    )

    messages_with_system = [{"type": "system", "content": system_message}] + state["messages"]

    response = llm.invoke(messages_with_system, config)
    return {"messages": response}

# === 4. Create the workflow graph ===
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("agent")

# === 5. Define graph edges ===
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

# === 6. Compile the workflow ===
agent = workflow.compile()
