import numpy as np
import cv2
import config

class Gaze:
    def __init__(self, display: bool = False):
        self.display = display
        pass

    def compute(self, frame: np.ndarray) -> float:

        print("Gaze compute")
        GazeFrame = frame.copy()
        
        if self.display:
            # Display the frame with gaze overlay (if needed)
            
            cv2.namedWindow("Gaze", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Gaze", config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
            cv2.putText(GazeFrame, "Gaze", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            cv2.imshow("Gaze", GazeFrame)
            pass
        return 0.0