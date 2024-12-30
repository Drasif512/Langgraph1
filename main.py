import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import START, StateGraph,END
from langgraph.prebuilt import tools_condition, ToolNode
from IPython.display import Image, display
import matplotlib.pyplot as plt
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from pprint import pprint
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LANGCHAIN_API_KEY= os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2=True
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_PROJECT="quickstart"
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=GEMINI_API_KEY,
    temperature=0
)


def multiply(a: int, b: int) -> int:
    """Multiply a and b.
    Args:
        a: first int
        b: second int
    """
    return print(a * b)

# This will be a tool
def add(a: int, b: int) -> int:
    """Adds a and b.
    Args:
        a: first int
        b: second int
    """
    return print(a + b)

def divide(a: int, b: int) -> float:
    """Divide a and b.
    Args:
        a: first int
        b: second int
    """
    return print(a / b)

tevily_search =  TavilySearchResults(max_results=3)
tools = [tevily_search, add, multiply, divide]
# print(tevily_search.invoke('How many revers are in punjab'))


class State(TypedDict):
    messages : Annotated[list, add_messages]

# #Object of graph

graph_builder = StateGraph(State)

llm_with_tools = llm.bind_tools(tools)


sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# Node
# def llm(state: State):
#     return {"messages": [llm.invoke(state["messages"])]}
def chatbot1(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}
def chatbot2(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}
def chatbot3(state: State):

    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot1", chatbot1)
tool_node = ToolNode(tools=[tevily_search, add, multiply, divide])
graph_builder.add_node('tools', tool_node)
graph_builder.add_conditional_edges("chatbot1",
                                    tools_condition,
                                    )
graph_builder.add_edge("tools", "chatbot1")
graph_builder.set_entry_point("chatbot1")
graph_builder.add_edge("chatbot1", END)

graph = graph_builder.compile()
memory = MemorySaver()
react_graph_memory = graph_builder.compile(checkpointer=memory)
thid : int = 1
config = {"configurable": {"thread_id": "thid"}}


messages = [HumanMessage(content=input("How i can help you :"))]


messages = react_graph_memory.invoke({"messages": messages}, config)


# for m in messages['messages']:
#     m.pretty_print()


# Generate the mermaid graph using graph.get_graph().draw_mermaid_png()
graph_data = graph.get_graph().draw_mermaid_png()  # Replace with appropriate method if necessary

with open("mermaid_graph.png", "wb") as f:
    f.write(graph_data)

print("Mermaid graph saved as mermaid_graph.png")



app = FastAPI()

# app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get('/')
def home():
    return {"message": "Welcome to the LangGraph!"}

# @app.post("/chat2")
# def chat2(input_message: str):
#     messages = [HumanMessage(content=input_message)]
#     response = llm.invoke( messages)
#     return {response.content}


# @app.post("/chat")
# def chat1(input_message: str):
#     messages = [HumanMessage(content=input_message)]
#     response = graph.invoke({'messages': messages})
#     return {"response": response["messages"][-1].content}

# @app.post("/chat2")
# def chat2(input_message: str):
#     messages = [HumanMessage(content=input_message)]
#     response = llm.invoke( messages)
#     return {response.content}



@app.post("/mchat")
def mchat1(input_message: str):
    messages = [HumanMessage(content=input_message)]
    response = react_graph_memory.invoke({"messages": messages},config)
    return{response["messages"][-1].content: add_messages}


