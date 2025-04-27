import numpy as np
import cv2
import config

class HeadPose:
    def __init__(self, display: bool = False):
        self.display = display
        pass

    def compute(self, frame: np.ndarray) -> float:

        print("HeadPose compute")
        HeadPoseFrame = frame.copy()

        if self.display:
            # Display the frame with head pose overlay (if needed)
            
            cv2.namedWindow("HeadPose", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("HeadPose", config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
            cv2.putText(HeadPoseFrame, "HeadPose", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            cv2.imshow("HeadPose", HeadPoseFrame)
            pass
        return 0.0