# -*- coding: utf-8 -*-

import numpy as np
import cv2
from .blink_counter_and_EAR_plot import *
import config
import time

class Blink:
    def __init__(self, display: bool = False):
        self.display = display

        self.blink_counter = BlinkCounterandEARPlot(
        video_path=0, # webcam
        threshold=0.294,
        consec_frames=3,
        save_video=False,
        )
        self.last_blink_time = None
        self.blink_intervals = []
        self.start_time = time.time()

    def compute(self, frame: np.ndarray) -> float:

        print("Blink compute")
        BlinkFrame = frame.copy()
        fps = 30.0
        # Process frame and get EAR
        frame, ear = self.blink_counter.process_frame(BlinkFrame)
        
        if ear is not None:
            prev_blink_count = self.blink_counter.blink_counter
            self.blink_counter._update_blink_detection(ear)
            now = time.time()
            
            # 깜빡임이 새로 발생된 경우
            if self.blink_counter.blink_counter > prev_blink_count:
                if self.last_blink_time is not None:
                    interval = now - self.last_blink_time
                    self.blink_intervals.append((now, interval))
                self.last_blink_time = now

            # 10초 기준으로 오래된 interval 제거
            self.blink_intervals = [(t, interval) for t, interval in self.blink_intervals if now - t <= 10.0]

            # 최근 10초간 평균 눈 감고 있던 시간 계산
            if self.blink_intervals:
                avg_interval = np.mean([interval for _, interval in self.blink_intervals])
                normalized_score = min(avg_interval / 10.0, 1.0)
            else :
                normalized_score = 0.0

            #self.blink_counter._update_visualization(frame, ear, fps)
            self.blink_counter._update_plot(ear)
        else :
            normalized_score = 0.0

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
        
        print("normalized_score : ", normalized_score)
        return normalized_score