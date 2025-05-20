from bleak import BleakClient
import asyncio
from enum import Enum

# Replace with your Hub's MAC address
SPIKE_HUB_MAC = "XX:XX:XX:XX:XX:XX"  
BLE_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"  # Nordic UART characteristic

class Command(Enum):
    DO_NOTHING = "DO NOTHING"
    ATTACK_1 = "ATTACK 1"
    ATTACK_2 = "ATTACK 2"

async def send_command(command: Command):
    async with BleakClient(SPIKE_HUB_MAC) as client:
        await client.write_gatt_char(BLE_UUID, command.value.encode())
        print(f"Sent: {command.value}")

# Example usage

if __name__ == "__main__":
    asyncio.run(send_command(Command.ATTACK_1))
