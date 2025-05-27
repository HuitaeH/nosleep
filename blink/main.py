# -*- coding: utf-8 -*-

import numpy as np
import cv2
from .blink_counter_and_EAR_plot import *
import config
import time

class Blink:
    def __init__(self, display: bool = False, display_graph: bool = False):
        self.display = display
        self.display_graph = display_graph

        self.blink_counter = BlinkCounterandEARPlot(
        video_path=0, # webcam
        threshold=0.294,
        consec_frames=1,
        save_video=False,
        )
        self.last_blink_time = None
        self.blink_intervals = []
        self.start_time = time.time()
        self.frame = None

        ## Calibration 
        self.button = True
        self.num_frame = 0
        self.default_interval = 0.0
        self.FRAME_THRESHOLD = 50 

    def compute(self, frame: np.ndarray) -> float:

        #print("Blink compute start")
        start_time = time.time()
        BlinkFrame = frame.copy()
        fps = 30.0
        # Process frame and get EAR
        frame, ear = self.blink_counter.process_frame(BlinkFrame)
        
        if ear is not None:
            prev_blink_count = self.blink_counter.blink_counter
            self.blink_counter._update_blink_detection(ear)
            now = time.time()
            
            ### modified 250527
            self.num_frame += 1
            if self.button and self.num_frame <= self.FRAME_THRESHOLD:
                if self.blink_counter.blink_counter > prev_blink_count:
                    if self.last_blink_time is not None:
                        interval = now - self.last_blink_time
                        self.blink_intervals.append((now, interval))
                    self.last_blink_time = now

                if self.num_frame == self.FRAME_THRESHOLD:
                    if self.blink_intervals:
                        self.default_interval = np.mean([interval for _, interval in self.blink_intervals])
                    else:
                        self.default_interval = 0.0
                    self.button = False  # calibration 종료
                    print(f"Default Interval: {self.default_interval:.2f} sec")

                normalized_score = 0.0  # calibration 중에는 점수 0
                self.blink_counter._update_plot(ear)

            else:
                if self.blink_counter.blink_counter > prev_blink_count:
                    if self.last_blink_time is not None:
                        interval = now - self.last_blink_time
                        self.blink_intervals.append((now, interval))
                    self.last_blink_time = now

                self.blink_intervals = [(t, interval) for t, interval in self.blink_intervals if now - t <= 10.0]

                if self.blink_intervals:
                    avg_interval = np.mean([interval for _, interval in self.blink_intervals])
                    # normalized_score = min(self.default_interval / avg_interval, 1.0)
                    normalized_score = min(max((avg_interval - self.default_interval)/10, 0.0), 1.0)

                else :
                    normalized_score = 0.0

                #self.blink_counter._update_visualization(frame, ear, fps)
                self.blink_counter._update_plot(ear)
        else :
            normalized_score = 0.0

        if self.display:
            # Display the frame with blink overlay (if needed)
            ## webcam 
            #cv2.namedWindow("Blink", cv2.WINDOW_NORMAL)
            #cv2.resizeWindow("Blink", config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
            cv2.putText(BlinkFrame, "Blink", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            #cv2.imshow("Blink", BlinkFrame)

            ## graph
            if self.display_graph:
                plot_img = self.blink_counter.plot_to_image()
                plot_img_resized = cv2.resize(
                    plot_img,
                    (config.WINDOW_WIDTH, config.WINDOW_HEIGHT),
                    interpolation=cv2.INTER_AREA
                )

                #cv2.imshow("Blink Plot", plot_img_resized)
                # (dtype) float → uint8
                if plot_img.dtype != np.uint8:
                    plot_img = np.clip(plot_img * 255, 0, 255).astype(np.uint8)

                # (채널) 그레이스케일 → BGR, RGBA → BGR
                if plot_img.ndim == 2:
                    plot_img = cv2.cvtColor(plot_img, cv2.COLOR_GRAY2BGR)
                elif plot_img.shape[2] == 4:
                    plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)

                # (색순서) RGB → BGR
                plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)

                # 3) 같은 크기로 리사이즈
                h, w = BlinkFrame.shape[:2]
                plot_img_resized = cv2.resize(
                    plot_img,
                    (w, h),
                    interpolation=cv2.INTER_AREA
                )

                # 4) 세로로 이어붙이기
                self.frame = cv2.vconcat([BlinkFrame, plot_img_resized])
            else:
                self.frame = BlinkFrame
                # 또는: combined = np.vstack((BlinkFrame, plot_img_resized))

                # 5) 하나의 창에 띄우기
                # cv2.namedWindow("Blink Combined", cv2.WINDOW_NORMAL)
                # cv2.resizeWindow("Blink Combined",  config.WINDOW_WIDTH, config.WINDOW_HEIGHT * 2)
                # cv2.imshow("Blink Combined", self.frame)
        #print("Blink compute end, time : ", time.time() - start_time)

        return 1-normalized_score

