import numpy as np
import cv2

class HeadPose:
    def __init__(self, display: bool = False):
        self.display = display
        pass

    def compute(self, frame: np.ndarray) -> float:

        print("HeadPose compute")

        if self.display:
            # Display the frame with head pose overlay (if needed)
            HeadPoseFrame = frame.copy()
            cv2.imshow("HeadPose", HeadPoseFrame)
            pass
        return 0.0