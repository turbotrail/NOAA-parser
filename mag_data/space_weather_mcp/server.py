import httpx
from typing import Optional, Any, Union
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("space-weather-portal")

BASE_URL = "https://www.ncei.noaa.gov/cloud-access/space-weather-portal/api/v1"

@mcp.tool()
async def get_satellites() -> dict[str, Any]:
    """
    Return a list of available satellites.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/satellites")
        response.raise_for_status()
        return response.json()

@mcp.tool()
async def get_instruments(sat: Optional[str] = None) -> dict[str, Any]:
    """
    Return a list of available instruments, optionally filtered by satellite.
    """
    params = {k: v for k, v in locals().items() if v is not None}
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/instruments", params=params)
        response.raise_for_status()
        return response.json()

@mcp.tool()
async def get_products(
    sat: Optional[str] = None,
    inst: Optional[str] = None,
    level: Optional[str] = None
) -> dict[str, Any]:
    """
    Return a list of available products, optionally filtered by satellite, instrument, or processing level.
    """
    params = {k: v for k, v in locals().items() if v is not None}
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/products", params=params)
        response.raise_for_status()
        return response.json()

@mcp.tool()
async def get_parameters(
    sat: Optional[str] = None,
    inst: Optional[str] = None,
    product: Optional[str] = None
) -> dict[str, Any]:
    """
    Return a list of available parameters, optionally filtered by satellite, instrument, or product.
    """
    params = {k: v for k, v in locals().items() if v is not None}
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/parameters", params=params)
        response.raise_for_status()
        return response.json()

@mcp.tool()
async def get_media_list(
    sat: Optional[str] = None,
    inst: Optional[str] = None,
    start: Optional[str] = None,
    stop: Optional[str] = None,
    page: Optional[int] = None,
    limit: Optional[int] = None,
    sort: Optional[str] = None,
    order: Optional[str] = None
) -> dict[str, Any]:
    """
    Return a list of available media metadata.
    """
    params = {k: v for k, v in locals().items() if v is not None}
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/media", params=params)
        response.raise_for_status()
        return response.json()

@mcp.tool()
async def get_media_names(
    sat: Optional[str] = None,
    inst: Optional[str] = None
) -> dict[str, Any]:
    """
    Return a distinct list of available media names.
    """
    params = {k: v for k, v in locals().items() if v is not None}
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/media/names", params=params)
        response.raise_for_status()
        return response.json()

@mcp.tool()
async def get_hapi_capabilities() -> dict[str, Any]:
    """
    Describes relevant implementation capabilities for this HAPI server.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/hapi/capabilities")
        response.raise_for_status()
        return response.json()

@mcp.tool()
async def get_hapi_about() -> dict[str, Any]:
    """
    Returns server identifier, contact information, and a brief description of the datasets served.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/hapi/about")
        response.raise_for_status()
        return response.json()

@mcp.tool()
async def get_hapi_catalog() -> dict[str, Any]:
    """
    Return list of datasets available from the HAPI server.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/hapi/catalog")
        response.raise_for_status()
        return response.json()

@mcp.tool()
async def get_hapi_info(
    dataset: str,
    parameters: Optional[str] = None
) -> dict[str, Any]:
    """
    Return list of all parameters for a given dataset.
    """
    params = {"dataset": dataset}
    if parameters:
        params["parameters"] = parameters
        
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/hapi/info", params=params)
        response.raise_for_status()
        return response.json()

@mcp.tool()
async def get_hapi_data(
    dataset: str,
    start: str,
    stop: str,
    parameters: Optional[str] = None,
    format: str = "json"
) -> Union[dict[str, Any], str]:
    """
    Returns table of requested parameter values based on provided conditions.
    format should be either 'json' or 'csv'. Default is 'json'.
    """
    params = {
        "dataset": dataset,
        "start": start,
        "stop": stop,
        "format": format
    }
    if parameters:
        params["parameters"] = parameters
        
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/hapi/data", params=params)
        response.raise_for_status()
        if format == "json":
            return response.json()
        return response.text

if __name__ == "__main__":
    mcp.run(transport='stdio')
