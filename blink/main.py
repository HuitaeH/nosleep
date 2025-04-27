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
            cv2.namedWindow("Blink", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Blink", 400, 300)
            cv2.putText(BlinkFrame, "Blink", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            cv2.imshow("Blink", BlinkFrame)
            pass

        return 0.0