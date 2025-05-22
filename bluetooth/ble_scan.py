import asyncio
from bleak import BleakScanner

from bleak import BleakClient
async def scan_ble_devices():
    devices = await BleakScanner.discover()
    for d in devices:
        print(f"Name: {d.name}, Address: {d.address}, RSSI: {d.rssi}")

asyncio.run(scan_ble_devices())


address = '10:D7:46:E4:F1:DF'
async def connect_and_interact():
    async with BleakClient(address) as client:
        if await client.is_connected():
            print(f"✅ Connected to {address}")
        else:
            print("❌ Connection failed")

asyncio.run(connect_and_interact())