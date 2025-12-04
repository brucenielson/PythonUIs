import chainlit as cl
import litellm

@cl.on_message
async def on_message(message: cl.Message):
  msg = cl.Message(content="")
  await msg.send()

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

  response = await litellm.acompletion(
    model="ollama/deepseek-r1:1.5b",
    messages = [
      system_message,
      {"role": "user", "content": message.content}
    ],
    api_base="http://localhost:11434",
    stream=True
  )

  async for chunk in response:
    if chunk:
      content = chunk.choices[0].delta.content
      if content:
        await msg.stream_token(content)

  await msg.update()
