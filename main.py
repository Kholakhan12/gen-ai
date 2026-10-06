from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

def get_llm():
    """Initializes and returns the ChatOllama model."""
    return ChatOllama(
        model="llama3.2:latest",
        temperature=0.7,
    )

def get_ai_response(messages, streaming=False):
    """
    Gets a response from the AI model.

    Args:
        messages (list): List of LangChain message objects.
        streaming (bool): Whether to stream the response.

    Returns:
        If streaming=False: The full response content.
        If streaming=True: A generator yielding chunks of the response.
    """
    llm = get_llm()
    if streaming:
        return llm.stream(messages)
    else:
        response = llm.invoke(messages)
        return response.content

if __name__ == "__main__":
    # Original CLI implementation kept for compatibility
    llm_instance = get_llm()
    chat_history = []

    print("Chat started! Type 'exit' to quit.\n")

    while True:
        question = input("You: ")

        if question.lower() in ["exit", "quit", "end"]:
            print("Chat ended.")
            break

        chat_history.append(HumanMessage(content=question))

        # Using the helper function for consistency
        response_content = get_ai_response(chat_history, streaming=False)

        chat_history.append(AIMessage(content=response_content))
        print("AI:", response_content)
