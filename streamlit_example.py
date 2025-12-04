# Run using: streamlit run streamlit_example.py
import streamlit as st
import ollama

st.set_page_config(page_title="DeepSeek R1 Chat", page_icon="🤖")


def convert_latex_delimiters(text):
    """Convert LaTeX delimiters from backslash-bracket to dollar signs"""
    # Replace display math delimiters
    text = text.replace(r'\[', '$$')
    text = text.replace(r'\]', '$$')
    # Replace inline math delimiters
    text = text.replace(r'\(', '$')
    text = text.replace(r'\)', '$')
    return text


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("thinking"):
            with st.expander("🧠 View Thinking Process", expanded=False):
                st.markdown(convert_latex_delimiters(msg["thinking"]))
        st.markdown(convert_latex_delimiters(msg["content"]))

# Chat input
user_input = st.chat_input("Ask DeepSeek R1")

if user_input:
    # Add and display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        thinking_display = st.empty()
        answer_display = st.empty()
        accumulated_thinking = ""
        accumulated_answer = ""

        with st.spinner("Thinking...", show_time=True):
            stream = ollama.chat(
                model="deepseek-r1:1.5b",
                messages=[{"role": m["role"], "content": m["content"]}
                          for m in st.session_state.messages],
                stream=True,
                think=True,
            )

            for chunk in stream:
                chunk_msg = chunk.get("message", {})

                # Stream thinking content
                if chunk_msg.get("thinking"):
                    accumulated_thinking += chunk_msg["thinking"]
                    with thinking_display.container():
                        with st.expander("🧠 View Thinking Process", expanded=True):
                            st.markdown(convert_latex_delimiters(accumulated_thinking) + "▌")

                # Stream answer content
                if chunk_msg.get("content"):
                    accumulated_answer += chunk_msg["content"]
                    answer_display.markdown(convert_latex_delimiters(accumulated_answer) + "▌")

        # Final display without cursor
        if accumulated_thinking:
            with thinking_display.container():
                with st.expander("🧠 View Thinking Process", expanded=False):
                    st.markdown(convert_latex_delimiters(accumulated_thinking))
        answer_display.markdown(convert_latex_delimiters(accumulated_answer))

        # Save to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": accumulated_answer,
            "thinking": accumulated_thinking or None
        })
