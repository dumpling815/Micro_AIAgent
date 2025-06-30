from LLMNode import LLMNode
from langserve import add_routes
from fastapi import FastAPI

endpoint_llm_node = LLMNode(
    name="llm-endpoint",
    mode=LLMNode.ENDPOINT,
    llm_config={
        "provider": "ollama",
        "model_name": "llama3",
        "temperature": 0.7,
        "base_url": "http://localhost:11434"
    }
)
router_llm_node = LLMNode(
    name="llm-router",
    mode=LLMNode.ROUTER,
    llm_config={
        "provider": "ollama",
        "model_name": "llama3",
        "temperature": 0.7,
        "base_url": "http://localhost:11434"
    }
)

endpoint_app = FastAPI()
add_routes(
    endpoint_app,
    path="/endpoint", # API endpoint
    runnable=endpoint_llm_node, # The LLMNode instance to handle requests
)
router_app = FastAPI()
add_routes(
    router_app,
    path="/router", # API endpoint
    runnable=router_llm_node, # The LLMNode instance to handle requests
)   