import bluetooth
import asyncio
from enum import Enum

class Command(Enum):
    DO_NOTHING = "DO NOTHING"
    ATTACK_1 = "0"
    ATTACK_2 = "1"

class BT:
    def __init__(self):
        self.sock = None
        self.connected_device = None
        self.connection = False

    def find_devices(self):
        devices = []
        nearby_devices = bluetooth.discover_devices(lookup_names=True)
        for addr, name in nearby_devices:
            devices.append({'name': name, 'addr': addr})
        return devices

    def select_and_connect(self):
        devices = self.find_devices()
        if not devices:
            print("No devices found.")
            return False
        
        for i, device in enumerate(devices):
            print(f"[{i}] {device['name']} ({device['addr']})")
        index = input("Select device (int, -1 : Not connect): ")
        
        try:
            if (int(index) == -1):
                self.connection = False
                return
            else:
                target = devices[int(index)]
                self.connection = True

        except (IndexError, ValueError):
            print("Invalid selection.")
            return False
        self.sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        try:
            self.sock.connect((target['addr'], 1))
            self.connected_device = target
            print(f"Connected to {target['name']} ({target['addr']})")
            return True
        except OSError:
            print("Connection failed.")
            return False

    async def send_command(self, command: Command):
        if not self.connection:
            return
        if not self.sock:
            print("Not connected to any device.")
            return

        if command == Command.DO_NOTHING:
            print("Doing nothing.")
            return

        try:
            self.sock.send(command.value)
            print(f"Sent command: {command.value}")
        except OSError:
            print("Failed to send command.")

    async def close(self):
        if self.sock:
            self.sock.close()
            print("Connection closed.")
            self.sock = None
            self.connected_device = None

# === Usage example ===

async def main():
    bt = BT()
    if not bt.select_and_connect():
        print("connection fail")
        return

    await bt.send_command(Command.ATTACK_1)
    await asyncio.sleep(1)
    await bt.send_command(Command.ATTACK_2)
    await asyncio.sleep(1)
    await bt.send_command(Command.DO_NOTHING)
    
    await bt.close()

if __name__ == "__main__":
    asyncio.run(main())
