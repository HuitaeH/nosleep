import asyncio
from bluetooth_client import BT, Command
import keyboard
import time




async def main():
    bt = BT()
    if not bt.select_and_connect():
        print("connection fail")
        return
    
    while True:
        if keyboard.is_pressed('1'):
            await bt.send_command(Command.ATTACK_1)
        elif keyboard.is_pressed('2'):
            await bt.send_command(Command.ATTACK_2)
        else:
            await bt.send_command(Command.DO_NOTHING)
        time.sleep(0.05)
        

asyncio.run(main())