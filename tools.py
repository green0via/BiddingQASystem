import os
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["ALL_PROXY"] = "socks5://127.0.0.1:7890"
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from langchain_community.utilities.serpapi import SerpAPIWrapper
# from models import gemini_temp
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

search = SerpAPIWrapper()

class AgentState(TypedDict):
    question: str
    answer: str
def call_tool(state):
    question = state["question"]
    result = search.run(question)
    return {"answer": result}

graph_builder = StateGraph(AgentState)
graph_builder.add_node("search", RunnableLambda(call_tool))
graph_builder.set_entry_point("search")
graph_builder.set_finish_point("search")

graph = graph_builder.compile()

if __name__ == '__main__':
    question = "今年的诺贝尔物理学奖得主是谁？"
    response = graph.invoke({"question": question})['answer']
    print(response)