# Run using: streamlit run streamlit_example.py
import streamlit as st
import ollama
import time

st.set_page_config(page_title="DeepSeek R1 Chat", page_icon="🤖")


def stream_data(text, delay: float = 0.02):
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
        # Show thinking if it exists
        if "thinking" in message and message["thinking"]:
            with st.expander("🧠 View Thinking Process", expanded=False):
                st.markdown(message["thinking"])
        st.markdown(message["content"])

if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display input prompt from user
    with st.chat_message("user"):
        st.markdown(prompt)

    # Processing
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        message_placeholder = st.empty()
        full_thinking = ""
        full_response = ""
        thinking_started = False
        response_started = False

        # Stream the response with a spinner while waiting for the initial response
        with st.spinner("Thinking...", show_time=True):
            response = ollama.chat(
                model="deepseek-r1:1.5b",
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                stream=True,
                think=True,  # This is the correct parameter!
            )

            # Process streaming chunks
            for chunk in response:
                # Capture thinking content
                if "message" in chunk and "thinking" in chunk["message"]:
                    thinking_content = chunk["message"]["thinking"]
                    if thinking_content:
                        full_thinking += thinking_content
                        if not thinking_started:
                            thinking_started = True
                        # Display thinking in an expander
                        with thinking_placeholder.container():
                            with st.expander("🧠 View Thinking Process", expanded=True):
                                st.markdown(full_thinking + "▌")

                # Capture regular content (final answer)
                if "message" in chunk and "content" in chunk["message"]:
                    content = chunk["message"]["content"]
                    if content:
                        full_response += content
                        if not response_started and thinking_started:
                            # Close the thinking expander when response starts
                            with thinking_placeholder.container():
                                with st.expander("🧠 View Thinking Process", expanded=False):
                                    st.markdown(full_thinking)
                            response_started = True
                        message_placeholder.markdown(full_response + "▌")

            # Final display without cursor
            if full_thinking:
                with thinking_placeholder.container():
                    with st.expander("🧠 View Thinking Process", expanded=False):
                        st.markdown(full_thinking)
            message_placeholder.markdown(full_response)

        # Add assistant response to chat history with thinking
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "thinking": full_thinking if full_thinking else None
        })