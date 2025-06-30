from abc import ABC, abstractmethod
from langchain_core.messages import BaseMessage
from typing import Optional, Union, Any
from enum import Enum

# Enum definitions for node types and modes
class NodeType(str, Enum):
    TOOL = "tool"
    LLM = "llm"

class LLMMode(str, Enum):
    ROUTER = "router"
    ENDPOINT = "endpoint"

class ToolMode(str, Enum):
    SEARCH = "search"
    IMAGE = "image"



# Base Node Class
class BaseNode(ABC):
    """A base class for all nodes in the agent graph."""
    def __init__(
            self, 
            name: str, 
            node_type: NodeType,
            mode: Union[LLMMode, ToolMode, str],
            description: Optional[str] = None,
            metadata: Optional[dict[str, Any]] = None,
    ):
        self.name = name
        self.node_type = node_type
        self.mode = mode
        self.description = description
        self.metadata = metadata if metadata is not None else {}

    @abstractmethod
    def __call__(self, input_message: list[BaseMessage]) -> list[BaseMessage]:
        """Method must be implemented by subclasses to process inputs."""
        pass

    def get_name(self) -> str:
        """Returns the name of the node."""
        return self.name
    
    def get_mode(self) -> str:
        """Returns the mode of the node."""
        return str(self.mode)
    
    def get_metadata(self) -> dict[str, Any]:
        """Returns the metadata of the node."""
        return self.metadata