import asyncio
import httpx

BASE_URL = "https://www.ncei.noaa.gov/cloud-access/space-weather-portal/api/v1"

async def main():
    async with httpx.AsyncClient() as client:
        print("--- /products ---")
        resp = await client.get(BASE_URL + "/products")
        print(resp.text[:500])
        print("--- /parameters ---")
        resp = await client.get(BASE_URL + "/parameters")
        print(resp.text[:500])

asyncio.run(main())
