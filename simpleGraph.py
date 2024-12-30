import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.graph.state import CompiledStateGraph
import matplotlib.pyplot as plt
from PIL import Image
from pprint import pprint


load_dotenv()
GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=GEMINI_API_KEY,
    temperature=0
)


# #class for stat

class State(TypedDict):
    messages : str

# #Object of graph

graph_builder = StateGraph(State)


# sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# Node functionality
def chatbot1(state: State):
    return llm.invoke(state['messages'])


# Node 
graph_builder.add_node("chatbot1", chatbot1)


graph_builder.set_entry_point("chatbot1")
graph_builder.add_edge("chatbot1", END)

graph = graph_builder.compile()


# # Generate the mermaid graph using graph.get_graph().draw_mermaid_png()
graph_data = graph.get_graph().draw_mermaid_png()  # Replace with appropriate method if necessary

with open("mermaid_graph.png", "wb") as f:
    f.write(graph_data)

print("Mermaid graph saved as mermaid_graph.png")

