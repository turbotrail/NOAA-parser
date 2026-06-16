# Space Weather Portal MCP Server

This is a Model Context Protocol (MCP) server that exposes the [NOAA Space Weather Portal API](https://www.ncei.noaa.gov/cloud-access/space-weather-portal/api/v1) to AI agents. It allows agents to seamlessly discover and query datasets, media, satellite telemetry, and instruments provided by NOAA.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (Python package manager)

## Installation

You do not need to manually install dependencies if you use `uv`, as it handles them on the fly. However, you can initialize the environment by running:

```bash
uv sync
```

## Available Tools

This server exposes the following tools to the AI agent:

### Core Data & HAPI Endpoints
- **`get_hapi_catalog`**: Get a list of all available datasets/products.
- **`get_hapi_info`**: Get the list of parameters for a specific dataset.
- **`get_hapi_data`**: Retrieve the actual tabular data (parameter values) for a specific dataset and time range.
- **`get_hapi_capabilities`**: Describe relevant HAPI server capabilities.
- **`get_hapi_about`**: Server identifier, contact info, and description.

### Media Endpoints
- **`get_media_list`**: Return a list of available media metadata (can filter by satellite, instrument, time range).
- **`get_media_names`**: Return a distinct list of available media names.

### Metadata Endpoints
- **`get_satellites`**: List all available satellites (e.g., DSCOVR, GOES-19, SOLAR-1, SWPC-Models).
- **`get_instruments`**: List all available instruments, optionally filtered by satellite.
- **`get_products`**: List all products, optionally filtered by satellite, instrument, or processing level.
- **`get_parameters`**: List all parameters, optionally filtered by satellite, instrument, or product.

## Usage

### Interactive Testing (MCP Inspector)

To interactively test the server's tools in your browser, use the official MCP Inspector. This is the recommended way to test if you're a developer making changes to the server.

```bash
npx -y @modelcontextprotocol/inspector uv run python server.py
```

This will spin up a local proxy and provide a `http://localhost:...` URL. Open that URL in your browser to test the tools.

### Using with an MCP Client (Claude Desktop, Cursor, etc.)

To give your AI agent access to these tools, you need to configure your agent's MCP settings to run this server. 

For example, in **Claude Desktop**, edit your `claude_desktop_config.json` (usually located at `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS) to include:

```json
{
  "mcpServers": {
    "space-weather": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/mag_data/space_weather_mcp",
        "run",
        "python",
        "server.py"
      ]
    }
  }
}
```

Make sure to replace `/absolute/path/to/mag_data/space_weather_mcp` with the actual absolute path to the directory containing `server.py`. Once configured, restart the client and the agent will automatically have access to query space weather data!
