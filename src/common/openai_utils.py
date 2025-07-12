import asyncio

async def to_event_stream(stream):
    """
    Converts an OpenAI stream to an event stream.
    """
    async for chunk in stream:
        yield f"data: {chunk.choices[0].delta.content or ''}\n\n"

