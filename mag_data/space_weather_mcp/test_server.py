import asyncio
from server import get_hapi_catalog

async def main():
    # the tool is actually registered via FastMCP, but we can't directly call it 
    # as an async function if it's wrapped. Wait, mcp.tool wraps it, but the original function is still callable or we can just import the original function.
    pass
