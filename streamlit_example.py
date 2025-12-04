import streamlit as st
import ollama
import time

def stream_data(text, delay: float=0.02):
    for word in text.split():
        yield word + " "
        time.sleep(delay)

# Input for the prompt
prompt = st.chat_input("Ask DeepSeek R1")

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display input prompt from user
    with st.chat_message("user"):
        st.markdown(prompt)

    # Processing
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Stream the response with a spinner while waiting for the initial response
        with st.spinner("Thinking...", show_time=True):
            response = ollama.chat(
                model="deepseek-r1:1.5b",
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                stream=True  # Enable streaming if supported by your ollama version
            )

            # If streaming is supported
            if hasattr(response, '__iter__'):
                for chunk in response:
                    if chunk and "message" in chunk and "content" in chunk["message"]:
                        content = chunk["message"]["content"]
                        full_response += content
                        message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            else:
                # Fallback for non-streaming response
                full_response = response["message"]["content"]
                # Simulate streaming for better UX
                for word in stream_data(full_response):
                    message_placeholder.markdown(full_response[:len(word)] + "▌")
                message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})