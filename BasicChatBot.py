import os,json
from typing import Annotated, TypedDict

# For LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

# For LanChain Runnable
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langchain_core.messages import ToolMessage, AIMessage

# Ollama 설정
# 기본적으로 ChatOllama는 로컬의 11434 포트에서 실행되는 ollama 서버에 연결.
# Ollama의 python SDK에 이에 관련된 코드가 작성되어 있음.
# 그리고 ChatOllama는 Ollama의 python SDK 클래스를 상속받아 구현되어 있음.
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0.0
)

# TavilySearch는 LangChain의 Search API를 사용하여 웹 검색을 수행하는 기능을 제공.
# LLM에 최적화된 검색 기능
os.environ["TAVILY_API_KEY"] = "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"  # Tavily API 키 설정
tavily = TavilySearch(max_results=2)
tools = [tavily]

# ChatOllama에 tavily를 포함한 tool 리스트를 바인딩.
llm_with_tools = llm.bind_tools(tools)


# LangGraph의 각 Node는 State를 가지는데, 이를 표현하기 위한 클래스.
# Typed Dict는 정적 타입 검사 지원
# Annotated는 타입 힌트에 메타데이터 추가 가능.
class State(TypedDict):
    messages: Annotated[list, add_messages]

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage"""
    def __init__(self, tools:list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}
    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(ToolMessage(content=json.dumps(tool_result),
                                       name=tool_call["name"],
                                       tool_call_id=tool_call["id"]))
        return {"messages": outputs}

# StateGraph를 통해, chatbot의 구조를 state machine 형태로 정의.
# Node를 추가함으로써 LLM과 Function을 표현.
# Edge를 통해 노드 간의 연결을 정의.
graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}
tool_node = BasicToolNode(tools=tools)


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)


def route_tools(state: State):
    """
    Use in the conditional edge to route to the ToolNode if the last message has tool calls.
    Otherwise, route to the end.
    """
    if isinstance(state,list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        print("I Think I should search...")
        return "my_tools"
    return END


graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function
    # But if you want to use node named something else apart from "tools"
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"my_tools": "tools", END: END}
)

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input), ("system", "Use tools only when user asks about real-time question.")]}):
        for value in event.values():
            print("Agent:", value["messages"][-1].content)


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

