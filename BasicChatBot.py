import os,json
from typing import Annotated, TypedDict

# For LangGraph
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

# For LanChain Runnable
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langchain_core.messages import ToolMessage

# Ollama 설정
# 기본적으로 ChatOllama는 로컬의 11434 포트에서 실행되는 ollama 서버에 연결.
# Ollama의 python SDK에 이에 관련된 코드가 작성되어 있음.
# 그리고 ChatOllama는 Ollama의 python SDK 클래스를 상속받아 구현되어 있음.
llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0.0
)

# TavilySearch는 LangChain의 Search API를 사용하여 웹 검색을 수행하는 기능을 제공.
# LLM에 최적화된 검색 기능
tavily = TavilySearch(max_results=2)
tools = [tavily]

# ChatOllama에 tavily를 포함한 tool 리스트를 바인딩.
llm_with_tools = llm.bind_tools(tools)


# LangGraphdml 각 Node는 State를 가지는데, 이를 표현하기 위한 클래스.
class State(TypedDict):
    messages: Annotated[list, add_messages]

# StateGraph를 통해, chatbot의 구조를 state machine 형태로 정의.
# Node를 추가함으로써 LLM과 Function을 표현.
# Edge를 통해 노드 간의 연결을 정의.
graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage"""


graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role":"user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


# 테스트 실행
while True:
    try:
        user_input = input("User:")
        if user_input.lower() in ["quit","exit","q"]:
            print("Bye")
            break
        stream_graph_updates(user_input)
    except KeyboardInterrupt:
        print("Bye")
        break

