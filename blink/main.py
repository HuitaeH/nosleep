import numpy as np
import cv2

class Blink:
    def __init__(self, display: bool = False):
        self.display = display
        pass

    def compute(self, frame: np.ndarray) -> float:

        print("Blink compute")

        if self.display:
            # Display the frame with blink overlay (if needed)
            BlinkFrame = frame.copy()
            cv2.imshow("Blink", BlinkFrame)
            pass

        return 0.0