# app.py
import asyncio
import chainlit as cl
import litellm

@cl.on_message
async def on_message(message: cl.Message):
    """
    Handles incoming messages from the user, sends them to the LLM,
    and streams the response back to the Chainlit interface.
    """
    # Create a placeholder message in the UI
    msg = cl.Message(content="Generating response...")
    await msg.send()

    # System prompt for the AI
    system_message = {
        "role": "system",
        "content": """You are an advanced AI assistant powered by the deepseek-r1:8b model.

Your strengths:
- Providing clear, accurate, and thoughtful responses
- Breaking down complex topics into understandable explanations
- Offering balanced perspectives on questions
- Being helpful while acknowledging your limitations

Guidelines:
- If you're uncertain about something, acknowledge it rather than making up information
- When appropriate, suggest related questions the user might want to ask
- Maintain a friendly, respectful tone
- Format your responses with markdown when it improves readability
"""
    }

    try:
        # Request completion from the model with streaming
        response = await litellm.acompletion(
            model="ollama/deepseek-r1:1.5b",
            messages=[
                system_message,
                {"role": "user", "content": message.content}
            ],
            api_base="http://localhost:11434",
            stream=True
        )
    except Exception as e:
        msg.content = f"Error generating response: {e}"
        await msg.update()
        return

    # Stream the response to the UI
    any_output = False
    async for chunk in response:
        if chunk and chunk.choices[0].delta.content:
            await msg.stream_token(chunk.choices[0].delta.content)
            any_output = True

    # Update the message after streaming completes
    if not any_output:
        msg.content = "Sorry, the model didn't generate a response."
        await msg.update()
    else:
        await msg.update()  # No need to change content here
