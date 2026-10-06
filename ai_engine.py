from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Union

class AIChatEngine:
    """Handles the AI logic separately from the UI."""
    def __init__(self, model_name: str = "llama3.2:latest", temperature: float = 0.7):
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
        )

    def get_response_stream(self, messages: List[Union[HumanMessage, AIMessage]]):
        """Streams the response from the LLM."""
        try:
            return self.llm.stream(messages)
        except Exception as e:
            raise RuntimeError(f"AI Engine Error: {str(e)}")

    def get_response(self, messages: List[Union[HumanMessage, AIMessage]]):
        """Returns a full response."""
        try:
            return self.llm.invoke(messages)
        except Exception as e:
            raise RuntimeError(f"AI Engine Error: {str(e)}")
