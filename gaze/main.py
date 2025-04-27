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
            cv2.namedWindow("Gaze", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Gaze", 400, 300)
            cv2.putText(GazeFrame, "Gaze", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            cv2.imshow("Gaze", GazeFrame)
            pass
        return 0.0