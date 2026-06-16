import asyncio
import httpx

BASE_URL = "https://www.ncei.noaa.gov/cloud-access/space-weather-portal/api/v1"

async def main():
    async with httpx.AsyncClient() as client:
        resp = await client.get(BASE_URL + "/instruments?sat=DSCOVR")
        print("/instruments?sat=DSCOVR:", resp.text)
        resp = await client.get(BASE_URL + "/products?sat=DSCOVR&inst=FC")
        print("/products?sat=DSCOVR&inst=FC:", len(resp.json().get("data", [])))

asyncio.run(main())
