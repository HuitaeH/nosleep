import bluetooth
class BT():
    def __init__(self):
        pass

    def find_devices(self):
        devices = []
        nearby_devices = bluetooth.discover_devices(lookup_names=True)
        for addr, name in nearby_devices:
            devices.append({'name' : name, 'addr' : addr})

        return devices
    
    def connect(self, addr):
        sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        try:
            sock.connect((addr, 1))
        except OSError:
            return None
        else:
            return sock
        
if __name__ == "__main__":
    bt = BT()
    devices = bt.find_devices()
    for device in devices:
        print(f"Name: {device['name']}, Address: {device['addr']}")
    
    # Example of connecting to a device
    if devices:
        sock = bt.connect(devices[0]['addr'])
        if sock:
            print(f"Connected to {devices[0]['name']}")
            sock.close()
        else:
            print("Failed to connect")