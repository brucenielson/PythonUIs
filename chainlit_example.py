# app.py
import chainlit as cl
import ollama


def convert_latex_delimiters(text):
    """Convert LaTeX delimiters from backslash-bracket to dollar signs"""
    if not text:
        return text
    # Replace display math delimiters
    text = text.replace(r'\[', '$$')
    text = text.replace(r'\]', '$$')
    # Replace inline math delimiters
    text = text.replace(r'\(', '$')
    text = text.replace(r'\)', '$')
    return text


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handles incoming messages from the user, sends them to the LLM,
    and streams the response back to the Chainlit interface with thinking process.
    """
    # System prompt for the AI
    system_message = {
        "role": "system",
        "content": """You are an advanced AI assistant powered by the deepseek-r1 model.

Guidelines:
- If you're uncertain about something, acknowledge it rather than making up information
- Format your responses with markdown when it improves readability
"""
    }

    # Create a step for thinking and messages for streaming
    thinking_step = cl.Step(name="💭 Thinking", type="tool")
    final_answer = cl.Message(content="")

    accumulated_thinking = ""
    accumulated_answer = ""

    try:
        # Request completion from the model with streaming and thinking enabled
        stream = ollama.chat(
            model="deepseek-r1:1.5b",
            messages=[
                system_message,
                {"role": "user", "content": message.content}
            ],
            stream=True,
            think=True,  # This is the critical parameter for Ollama native API
        )

        thinking_started = False
        answer_started = False
        answer_buffer = ""

        # Stream the response to the UI
        for chunk in stream:
            chunk_msg = chunk.get("message", {})

            # Handle thinking content
            if chunk_msg.get("thinking"):
                if not thinking_started:
                    thinking_started = True
                    await thinking_step.send()

                thinking_text = chunk_msg["thinking"]
                accumulated_thinking += thinking_text
                thinking_step.output = convert_latex_delimiters(accumulated_thinking)
                await thinking_step.update()

            # Handle answer content
            if chunk_msg.get("content"):
                if not answer_started:
                    answer_started = True
                    if thinking_started:
                        # Finalize the thinking step
                        await thinking_step.update()
                    await final_answer.send()

                answer_text = chunk_msg["content"]
                accumulated_answer += answer_text
                answer_buffer += answer_text

                # Only update every 10 characters or so to avoid overwhelming the socket
                if len(answer_buffer) >= 10:
                    await final_answer.stream_token(answer_buffer)
                    answer_buffer = ""

        # Send any remaining buffered content
        if answer_buffer:
            await final_answer.stream_token(answer_buffer)

        # Update final answer with LaTeX conversion
        final_answer.content = convert_latex_delimiters(accumulated_answer)
        await final_answer.update()

    except Exception as e:
        error_msg = cl.Message(content=f"❌ Error generating response: {str(e)}")
        await error_msg.send()


@cl.on_chat_start
async def start():
    """
    Sends a welcome message when the chat starts.
    """
    await cl.Message(
        content="👋 Hello! I'm powered by **DeepSeek R1**. I'll show you my thinking process before answering.\n\n"
                "Try asking me a math problem or reasoning question!"
    ).send()

