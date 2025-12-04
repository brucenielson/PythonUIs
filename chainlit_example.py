# app.py
import asyncio
import chainlit as cl
from openai import AsyncOpenAI
import time

# Use OpenAI client pointing to Ollama
client = AsyncOpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1/"
)


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handles incoming messages from the user, sends them to the LLM,
    and streams the response back to the Chainlit interface with thinking process.
    """
    start_time = time.time()

    # System prompt for the AI
    system_message = {
        "role": "system",
        "content": """You are an advanced AI assistant powered by the deepseek-r1 model.

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
        stream = await client.chat.completions.create(
            model="deepseek-r1:1.5b",
            messages=[
                system_message,
                {"role": "user", "content": message.content}
            ],
            stream=True
        )
    except Exception as e:
        await cl.Message(content=f"Error generating response: {e}").send()
        return

    # Track whether we're in thinking mode
    thinking = False

    # Create a step for thinking and a message for final answer
    thinking_step = cl.Step(name="Thinking")
    final_answer = cl.Message(content="")

    # Stream the response to the UI
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content

            # Check for thinking tags
            if content == "<think>":
                thinking = True
                await thinking_step.send()
                continue

            if content == "</think>":
                thinking = False
                thought_duration = round(time.time() - start_time)
                thinking_step.name = f"Thought for {thought_duration}s"
                await thinking_step.update()
                continue

            # Stream to appropriate destination
            if thinking:
                await thinking_step.stream_token(content)
            else:
                await final_answer.stream_token(content)

    # Send the final answer
    await final_answer.send()


@cl.on_chat_start
async def start():
    """
    Sends a welcome message when the chat starts.
    """
    await cl.Message(
        content="Hello! I'm powered by DeepSeek R1. I'll show you my thinking process before answering. Ask me anything!"
    ).send()
