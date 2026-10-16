import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PYTHON_BIN = sys.executable
SERVER_PATH = Path(__file__).parent / "dscr_mcp_server.py"  # generated file

USE_MCP = True
if USE_MCP:
    async def _list_dscr_tools():
        params = StdioServerParameters(command=PYTHON_BIN, args=[str(SERVER_PATH)])
        # errlog=sys.stderr so server-side errors (e.g. missing fastmcp) are visible
        async with stdio_client(params, errlog=sys.stderr) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                return [t.name for t in tools.tools]

    tool_names = asyncio.run(_list_dscr_tools())
    assert tool_names == ["calculate_dscr"], tool_names
    print("dscr_mcp_server.py: OK -> advertises tools:", tool_names)
else:
    print("USE_MCP is False -- skipped (no server was built).")