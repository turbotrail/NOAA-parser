import asyncio
import httpx

BASE_URL = "https://www.ncei.noaa.gov/cloud-access/space-weather-portal/api/v1"

endpoints = [
    "/satellites", "/instruments", "/products", "/datasets", "/data", "/metadata", 
    "/observations", "/models", "/parameters", "/catalog", "/info"
]

async def main():
    async with httpx.AsyncClient() as client:
        for ep in endpoints:
            resp = await client.get(BASE_URL + ep)
            print(f"{ep}: {resp.status_code}")

asyncio.run(main())
