import numpy as np
import cv2

class Gaze:
    def __init__(self, display: bool = False):
        self.display = display
        pass

    def compute(self, frame: np.ndarray) -> float:

        print("Gaze compute")
        
        if self.display:
            # Display the frame with gaze overlay (if needed)
            GazeFrame = frame.copy()
            cv2.imshow("Gaze", GazeFrame)
            pass
        return 0.0