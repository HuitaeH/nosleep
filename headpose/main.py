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
            cv2.namedWindow("HeadPose", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("HeadPose", 400, 300)
            cv2.putText(HeadPoseFrame, "HeadPose", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            cv2.imshow("HeadPose", HeadPoseFrame)
            pass
        return 0.0