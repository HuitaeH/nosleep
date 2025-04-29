import numpy as np
import cv2
from .blink_counter_and_EAR_plot import *
import config

class Blink:
    def __init__(self, display: bool = False):
        self.display = display

        self.blink_counter = BlinkCounterandEARPlot(
        video_path=0, # webcam
        threshold=0.294,
        consec_frames=3,
        save_video=False,
        )
    

    def compute(self, frame: np.ndarray) -> float:

        print("Blink compute")
        BlinkFrame = frame.copy()
        fps = 30.0
        # Process frame and get EAR
        frame, ear = self.blink_counter.process_frame(BlinkFrame)
        
        if ear is not None:
            self.blink_counter._update_blink_detection(ear)
            #self.blink_counter._update_visualization(frame, ear, fps)
            self.blink_counter._update_plot(ear)

        if self.display:
            # Display the frame with blink overlay (if needed)
            ## webcam 
            cv2.namedWindow("Blink", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Blink", config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
            cv2.putText(BlinkFrame, "Blink", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            cv2.imshow("Blink", BlinkFrame)

            ## graph
            plot_img = self.blink_counter.plot_to_image()
            plot_img_resized = cv2.resize(
                plot_img,
                (config.WINDOW_WIDTH, config.WINDOW_HEIGHT),
                interpolation=cv2.INTER_AREA
            )
            cv2.imshow("Blink Plot", plot_img_resized)
            pass

        return 0.0