from BaseNode import BaseNode
from typing import Dict, Any, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
import requests

class LLMNode(BaseNode):
    """A node that represents an LLM (Large Language Model) in the state graph."""
    def __init__(self, name: str, model: str, description: Optional[str] = None):
        super().__init__(name, "LLM", description)
        self.llm = model

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process the input and return the output from the LLM."""
        if "messages" not in inputs:
            raise ValueError("There's no messages in input.")
        
        messages = inputs["messages"]
        response = self.model.invoke(messages)
        
        return {"messages": response}