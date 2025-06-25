# Base Node Class
from typing import TypedDict, Annotated, Optional

class BaseNode:
    """A base class for all nodes in the state graph."""
    def __init__(self, name: str, node_type: str, description: Optional[str] = None):
        self.name = name
        self.node_type = node_type
        self.description = description

    def __call__(self, inputs: dict):
        """Method to be implemented by subclasses to process inputs."""
        raise NotImplementedError("Subclasses should implement this method.")

    def get_name(self) -> str:
        """Returns the name of the node."""
        return self.name