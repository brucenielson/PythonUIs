import streamlit as st
import requests
import json

st.title("Streamlit + Ollama Streaming (with <think> tags)")

model = st.selectbox(
    "Model:",
    ["deepseek-r1:1.5b", "qwen2.5:1.5b"],
)

prompt = st.text_area("Prompt:", height=150)

if st.button("Run (Stream)"):
    if not prompt.strip():
        st.error("Enter a prompt.")
    else:
        st.write("### Output (streaming):")

        # Output placeholder
        output_area = st.empty()
        collected_text = ""

        # Send streaming request to Ollama
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True
        }
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            stream=True
        )

        # Read streaming chunks
        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line.decode("utf-8"))
            chunk = data.get("response", "")
            collected_text += chunk
            output_area.code(collected_text)

        st.success("Done.")
