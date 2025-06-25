from basenode import BaseNode

class RouterNode(BaseNode):
    def __init__(self, name: str, description: str = None):
        super().__init__(name, "Router", description)
        self.routes = {}