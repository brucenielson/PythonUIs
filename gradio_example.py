import gradio as gr
import google.generativeai as genai

# ---------------------------
# Secret loading
# ---------------------------
def get_secret(secret_file: str) -> str:
    try:
        with open(secret_file, 'r') as file:
            secret_text: str = file.read().strip()
    except FileNotFoundError:
        print(f"The file '{secret_file}' does not exist.")
        secret_text = ""
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e

    return secret_text


# Load Gemini API key
gemini_secret: str = get_secret(r"D:\Documents\Secrets\gemini_secret.txt")
genai.configure(api_key=gemini_secret)


# ---------------------------
# Chat handler
# ---------------------------
def chat_with_gemini(message, chat_history, system_message, model_name, temperature, max_tokens):
    if chat_history is None:
        chat_history = []

    # Convert Gradio history → Gemini history format
    gemini_history = []
    if system_message.strip():
        gemini_history.append({"role": "user", "parts": [f"System message: {system_message}"]})

    for msg in chat_history:
        role = msg["role"]
        content = msg["content"]

        gemini_history.append({
            "role": "user" if role == "user" else "model",
            "parts": [content]
        })

    # Append latest user message
    gemini_history.append({"role": "user", "parts": [message]})

    # Create model
    model = genai.GenerativeModel(model_name)

    # Generate reply
    response = model.generate_content(
        gemini_history,
        generation_config=genai.types.GenerationConfig(
            temperature=float(temperature),
            max_output_tokens=int(max_tokens)
        )
    )

    bot_reply = response.text

    # Append to UI chat
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": bot_reply})

    return chat_history, ""


# ---------------------------
# Gradio UI
# ---------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🌟 Gemini Chat Interface")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(type="messages", elem_id="chatbot", height=500)
            msg = gr.Textbox(placeholder="Send a message...", show_label=False)

            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear", variant="secondary")

        with gr.Column(scale=1):
            gr.Markdown("### Model Settings")

            model_dropdown = gr.Dropdown(
                choices=["gemini-2.0-flash", "gemini-2.0-pro", "gemini-1.5-flash"],
                value="gemini-2.0-flash",
                label="Model"
            )

            system_message = gr.Textbox(
                placeholder="You are a helpful AI assistant...",
                label="System Message",
                lines=4
            )

            with gr.Accordion("Advanced Settings", open=False):
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.01,
                    label="Temperature"
                )

                max_tokens = gr.Slider(
                    minimum=100,
                    maximum=4096,
                    value=512,
                    step=50,
                    label="Max Tokens"
                )

    # Wire events
    submit_btn.click(
        chat_with_gemini,
        inputs=[msg, chatbot, system_message, model_dropdown, temperature, max_tokens],
        outputs=[chatbot, msg],
    )

    msg.submit(
        chat_with_gemini,
        inputs=[msg, chatbot, system_message, model_dropdown, temperature, max_tokens],
        outputs=[chatbot, msg],
    )

    clear_btn.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    demo.launch()
