from BaseNode import BaseNode, NodeType, LLMMode, ToolMode
from typing import Dict, Any, Optional, Literal, Union

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage, ToolCall
from langchain_core.runnables import Runnable
from langchain.schema.runnable import Runnable
from langchain_core.prompts import ChatPromptTemplate

from langserve import RemoteRunnable, add_routes
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
import requests, os

"""
LLM config example:
llm_config = {
    "provider": "openai", # or "ollama", "local", etc.
    "model": "gpt-3.5-turbo", # or "llama3", any other model
    "temperature": 0.7, # optional, default is 0.7
    "api_key": "env:OPENAI_API_KEY", # optional, if not provided, will use the default environment variable
    "base_url": "http://localhost:11434" # optional, for ollama or local models
}
"""

class LLMNode(BaseNode):
    """A node that represents an LLM (Large Language Model) in the state graph."""
    def __init__(
            self, 
            name: str, 
            mode: LLMMode,
            llm_config: dict,
            tool_config: Optional[dict] = None,
            description: Optional[str] = None,
            metadata: Optional[dict[str, Any]] = {"Intensive": "GPU"},
    ):
        super().__init__(
            name=name,
            node_type=NodeType.LLM,
            mode=mode,
            description=description,
            metadata=metadata,
        )
        self.llm: Runnable = self._load_llm(llm_config)
        self.tools: Dict[str, Runnable] = self._available_tools(tool_config)  # Dictionary to hold tools if needed
    
    def _load_llm(self, llm_config: dict) -> Runnable:
        # Load the LLM based on the provided configuration.
        provider = llm_config.get("provider").lower()
        model = llm_config.get("model")
        temperature = llm_config.get("temperature", 0.7) # Default Temperature: 0.7

        if provider == "openai":
            api_key = llm_config.get("api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key is not setted")
            return ChatOpenAI(model=model, temperature=temperature, openai_api_key=api_key)
        
        elif provider == "ollama":
            base_url = llm_config.get("base_url", "http://localhost:11434")
            return ChatOllama(model=model, temperature=temperature, base_url=base_url)

        else:
            # In this project, we only support OpenAI and Ollama.
            # If you want to support more providers, you can add them here.
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def _available_tools(self, tool_config: Optional[dict]) -> list[str]:
        """Check available tools based on the tool configuration."""
        if not tool_config:
            return []
        available_tools = []
        for tool_name, tool_details in tool_config.items():
            if tool_details.get("enabled", True):
                available_tools.append(tool_name)
        return available_tools
    

    async def __call__(self, input_messages: list[BaseMessage]) -> list[BaseMessage]:
        """Process the input and return the output from the LLM."""
        user_msg = next((msg for msg in reversed(input_messages) if isinstance(msg, HumanMessage)), None)
        if user_msg is None:
            return ValueError("No user message found in input messages.")
        if self.mode == LLMMode.ROUTER:
            last_msg = input_messages[-1] if input_messages else None
            if isinstance(last_msg, HumanMessage):
                # Case 1: First user input -> Tool call choice
                # Generating a routing decision using the LLM
                routing_prompt = ChatPromptTemplate.from_messages([
                    SystemMessage(content="You are a routing agent. Route the input to the appropriate LLM based on the content."),
                    HumanMessage(content="Look at the following input and decide which tool you should use. If you don't need to use any tool, just return 'llm'.\n\n{user_msg}"),
                ])
                routing_chain = routing_prompt | self.llm

                # Get the routing decision from the LLM
                decision_msg = await routing_chain.ainvoke(routing_prompt)
                if not isinstance(decision_msg, str):
                    raise ValueError("Routing decision must be a string.")
                decision = decision_msg.content.strip().lower()

                # Branch based on the routing decision
                if decision in self.tools:
                    return [AIMessage(
                        content = None,
                        tool_calls=[ToolCall(
                            name = decision,
                            args = {"query": last_msg.content}
                        )]
                    )]
                elif decision == "llm":
                    return [last_msg]
                else:
                    raise ValueError(f"Invalid routing decision: {decision}. Available tools: {', '.join(self.tools.keys())} or 'llm'.")
            elif isinstance(last_msg, ToolMessage):
                # Case 2: After tool call -> Branching based on tool output
                tool_result = last_msg.content
                tool_id = last_msg.tool_call_id

                routing_followup_prompt = ChatPromptTemplate.from_messages([
                    SystemMessage(content="You are a routing agent. Based on the tool output, decide the next action."),
                    HumanMessage(content=(
                        f"Look at the following tool output and decide what to do next.\n\n{tool_result}"
                        f"\n\nIf you want to call another tool, return the tool name.\n "
                        f"Available tools: {', '.join(self.tools.keys())} or 'llm' if you want to continue with the LLM."
                    ))
                ])

                decision_msg = await self.llm.ainvoke(routing_followup_prompt)
                decision = decision_msg.content.strip().lower()
                if decision in self.tools:
                    return [AIMessage(
                        content = None,
                        tool_calls=[ToolCall(
                            name = decision,
                            args = {"query": tool_result}
                        )]
                    )]
                elif decision == "llm":
                    return [AIMessage(content=tool_result)]
                else:
                    raise ValueError(f"Invalid routing decision: {decision}. Available tools: {', '.join(self.tools.keys())} or 'llm'.")
            else:
                # Case 3: Invalid input message type - Router only supports HumanMessage or ToolMessage
                raise ValueError("Invalid input message type for routing mode.")
            
        else: # LLMMode.ENDPOINT
            response = await self.llm.ainvoke(input_messages)
            return response if isinstance(response, list) else [response]
