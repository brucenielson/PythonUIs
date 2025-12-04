# Run using: streamlit run streamlit_example.py
import streamlit as st
import ollama

st.set_page_config(page_title="DeepSeek R1 Chat", page_icon="🤖")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and message.get("thinking"):
            with st.expander("🧠 View Thinking Process", expanded=False):
                st.markdown(message["thinking"])
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Ask DeepSeek R1")

if prompt:
    # Add and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        message_placeholder = st.empty()
        full_thinking = ""
        full_response = ""

        with st.spinner("Thinking...", show_time=True):
            response = ollama.chat(
                model="deepseek-r1:1.5b",
                messages=[{"role": m["role"], "content": m["content"]}
                          for m in st.session_state.messages],
                stream=True,
                think=True,
            )

            for chunk in response:
                msg = chunk.get("message", {})

                # Stream thinking content
                if msg.get("thinking"):
                    full_thinking += msg["thinking"]
                    with thinking_placeholder.container():
                        with st.expander("🧠 View Thinking Process", expanded=True):
                            st.markdown(full_thinking + "▌")

                # Stream response content
                if msg.get("content"):
                    full_response += msg["content"]
                    message_placeholder.markdown(full_response + "▌")

        # Final display without cursor
        if full_thinking:
            with thinking_placeholder.container():
                with st.expander("🧠 View Thinking Process", expanded=False):
                    st.markdown(full_thinking)
        message_placeholder.markdown(full_response)

        # Save to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "thinking": full_thinking or None
        })
