import gradio as gr


def chat_with_grok(message, chat_history, system_message, model_name, temperature, max_tokens):
    # Ensure history is a list of message dicts
    if chat_history is None:
        chat_history = []
    # Add user message (Gradio 4.x expects dicts with role/content)
    chat_history.append({"role": "user", "content": message})

    # Create bot response (replace this with actual API call)
    bot_message = (
        f"You selected model: {model_name}\n"
        f"System message: {system_message}\n"
        f"Temperature: {temperature}\n"
        f"Max tokens: {max_tokens}\n\n"
        f"Your message: {message}"
    )

    # Add assistant message
    chat_history.append({"role": "assistant", "content": bot_message})

    # Return updated history AND an empty string to clear the input textbox
    return chat_history, ""


with gr.Blocks() as demo:
    gr.Markdown("# Grok AI Chat Interface")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(elem_id="chatbot", height=400)  # value is list of dicts
            msg = gr.Textbox(placeholder="Send a message...", show_label=False)

            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear", variant="secondary")

        with gr.Column(scale=1):
            gr.Markdown("### Model Settings")

            model_dropdown = gr.Dropdown(
                choices=["grok-1", "grok-2", "grok-3-beta"],
                value="grok-3-beta",
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
                    maximum=4000,
                    value=1000,
                    step=100,
                    label="Max Tokens"
                )

    # Wire up events: outputs are (chatbot, msg) so textbox is cleared
    submit_btn.click(
        chat_with_grok,
        inputs=[msg, chatbot, system_message, model_dropdown, temperature, max_tokens],
        outputs=[chatbot, msg],
    )

    msg.submit(
        chat_with_grok,
        inputs=[msg, chatbot, system_message, model_dropdown, temperature, max_tokens],
        outputs=[chatbot, msg],
    )

    clear_btn.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    demo.launch()
