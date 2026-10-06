import streamlit as st
from main import get_ai_response
from langchain_core.messages import HumanMessage, AIMessage

# Page configuration for a professional look
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stChatInputContainer {
        padding-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session():
    """Initialize session state for chat history."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

def main():
    initialize_session()

    # --- Sidebar ---
    with st.sidebar:
        st.title("⚙️ Settings")
        st.markdown("---")
        st.info("This AI app is powered by Llama 3.2 via Ollama and LangChain.")

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # --- Main UI ---
    st.title("🤖 AI Assistant")
    st.caption("A modern interface for your custom LLM logic")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

    # React to user input
    if prompt := st.chat_input("Type your message here..."):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add user message to chat history
        st.session_state.messages.append(HumanMessage(content=prompt))

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                with st.spinner("Thinking..."):
                    # Call the logic from main.py with streaming enabled
                    response_stream = get_ai_response(st.session_state.messages, streaming=True)

                    for chunk in response_stream:
                        full_response += chunk.content
                        # Update the placeholder with the current response text
                        message_placeholder.markdown(full_response + "▌")

                    # Final update to remove the cursor
                    message_placeholder.markdown(full_response)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                full_response = "Sorry, I encountered an error while processing your request."
                message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append(AIMessage(content=full_response))

if __name__ == "__main__":
    main()
