import setup_path
import airsim

import numpy as np
import os
import tempfile
import pprint
import cv2

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

state = client.getMultirotorState()
s = pprint.pformat(state)
print("state: %s" % s)


state = client.getMultirotorState()
print("state: %s" % pprint.pformat(state))

client.moveToPositionAsync(10, 0, -10, 5).join()
client.moveToPositionAsync(10, 10, -10, 5).join()
client.moveToPositionAsync(0, 10, -10, 5).join()
client.moveToPositionAsync(0, 0, -10, 5).join()



print("Resetting to original state...")
client.reset()
client.armDisarm(False)

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False)
