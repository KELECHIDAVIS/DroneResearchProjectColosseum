# simple_test.py
import airsim
import time

print("Connecting...")
client = airsim.MultirotorClient()
client.confirmConnection()

client.enableApiControl(True)
client.armDisarm(True)

print("Taking off...")
client.takeoffAsync().join()

print("Flying in a square pattern...")
client.moveToPositionAsync(5, 0, -5, 2).join()
client.moveToPositionAsync(5, 5, -5, 2).join()
client.moveToPositionAsync(0, 5, -5, 2).join()
client.moveToPositionAsync(0, 0, -5, 2).join()

print("Landing...")
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)

print("Done!")