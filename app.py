from typing import Annotated
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# Load environment variables
load_dotenv()

# Get API key from environment variable
gemini_api_key = os.getenv('GEMINI_API_KEY')

class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatGoogleGenerativeAI( 
    model="gemini-1.5-flash",
    max_retries=2,
    google_api_key=gemini_api_key
)
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile()

####################################################################################################
####################################################################################################
# Save the graph visualization as a PNG file
try:
    mermaid_png = graph.get_graph().draw_mermaid_png()
    with open("graph_visualization.png", "wb") as f:
        f.write(mermaid_png)
    print("Graph visualization saved as 'graph_visualization.png'")
except Exception as e:
    print(f"Could not generate graph visualization: {str(e)}")

for event in graph.stream({"messages": [("user", "I am Talha")]}):
    print("EVENT:", list(event.values())[0]["messages"][-1].content)

# Stream graph updates based on user input
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)
# Stream graph updates based on user input
while True: 
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break