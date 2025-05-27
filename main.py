
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from headpose.main import HeadPose
from gaze.main import Gaze
from blink.main import Blink
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import config
import keyboard  # pip install keyboard
import sys
import platform
import time


from spike_tx import BT, Command
import asyncio
from predictor import RealtimePredictor

DISPLAY = True             # camera display
DISPLAY_GRAPH = False       # graph display
DISPLAY_OVERALL = False     # overall concentration display
rnn_model_path = './models/drowsiness_rnn_best.h5'


def main():

    print("Initializing modules...")
    cap = None
    picam = None
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(0)
            # resolution settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        from picamera.array import PiRGBArray
        from picamera import PiCamera
        picam = PiCamera()
        picam.resolution = (640, 480)
        picam.framerate = 30
        rawCapture = PiRGBArray(picam, size=(640, 480))
        time.sleep(0.1)  # 카메라 워밍업
        camera_type = "picamera"

    #for bluetooth
    bt = BT()
    bt.select_and_connect()
        


    hp = HeadPose(display=DISPLAY, display_graph = DISPLAY_GRAPH, frame_width=640, frame_height=480)
    gz = Gaze(display=DISPLAY, display_graph = DISPLAY_GRAPH)
    bk = Blink(display=DISPLAY, display_graph = DISPLAY_GRAPH)
    graph = ConcentrationGraph()

    #RNN
    predictor = RealtimePredictor(model_path = rnn_model_path) # model path 
    
    print("Modules initialized.")

        # 2. 캘리브레이션 시작
    frame_id = 0
    print("Start calibration. Press 'c' to start.")
    empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(empty_frame, "Press 'c' to start calibration",
                (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    while True:
        cv2.imshow("Calibration", empty_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            print("Calibration started.")
            break

    cv2.destroyWindow("Calibration")
    while True:
        frame = None
        if (not hp.button and not gz.button) :
            break
        if platform.system() == "Windows":
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera.")
                break
        else:
            capture = next(picam.capture_continuous(rawCapture, format="bgr", use_video_port=True))
            frame = capture.array
            rawCapture.truncate(0)

        head_score, pitch = hp.compute(frame)
        gaze_score = gz.compute(frame)
        blink_score = bk.compute(frame)

        print(f"[{frame_id}] H: {head_score:.2f}, G: {gaze_score:.2f}, B: {blink_score:.2f}")

        frame_id += 1
        if DISPLAY:
            # 2) 가로로 이어붙이기
            all_combined = cv2.hconcat([hp.frame,
                                        gz.frame,
                                        bk.frame])
            # (또는 np.hstack([…, …, …]) 사용 가능)

            # 3) 창 띄우기
            cv2.namedWindow("All Combined", cv2.WINDOW_NORMAL)
            # 가로 너비 3×W, 세로 높이 2×H 로 리사이즈
            cv2.resizeWindow("All Combined",
                            config.WINDOW_WIDTH * 3,
                            config.WINDOW_HEIGHT * 2 if DISPLAY_GRAPH else config.WINDOW_HEIGHT)
            cv2.putText(all_combined, "Calibrating",
                        ((all_combined.shape[1] - cv2.getTextSize("Calibrating", cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)[0][0]) // 2,
                         (all_combined.shape[0] + cv2.getTextSize("Calibrating", cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)[0][1]) // 2),
                         cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 5, cv2.LINE_AA)


            cv2.imshow("All Combined", all_combined)

            #cv2.imshow("Frame", frame)
        cv2.waitKey(1)


    print("Calibration done.")

    while True:
        frame = None
        if platform.system() == "Windows":
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera.")
                break
        else:
            capture = next(picam.capture_continuous(rawCapture, format="bgr", use_video_port=True))
            frame = capture.array
            rawCapture.truncate(0)

        # ?�� 모듈 모두 0~1 ?��코어 반환
        head_score, pitch = hp.compute(frame)
        bk.blink_counter.set_pitch(pitch)
        gaze_score = gz.compute(frame)
        blink_score = bk.compute(frame)

        # rnn model inference
        predictor.update(head_score, gaze_score, blink_score)
        pred_class = predictor.predict_if_ready()
        if pred_class is not None:
            print(f"prediction: {pred_class}")  

        # TODO : should be replaced with a model
        result = Command.DO_NOTHING

        ## send to robot
        asyncio.run(bt.send_command(result))

        # 전체 집중도 예시 (가중 평균)
        # overall = (head_score*0.3 + gaze_score*0.3 + blink_score*0.4)
        if (DISPLAY_OVERALL):
            graph._update_plot(0, gaze_score, blink_score, head_score)
        print(f"H: {head_score:.2f}, G: {gaze_score:.2f}, B: {blink_score:.2f}")
        if DISPLAY:
            # 2) 가로로 이어붙이기
            all_combined = cv2.hconcat([hp.frame,
                                        gz.frame,
                                        bk.frame])
            # (또는 np.hstack([…, …, …]) 사용 가능)

            # 3) 창 띄우기
            cv2.namedWindow("All Combined", cv2.WINDOW_NORMAL)
            # 가로 너비 3×W, 세로 높이 2×H 로 리사이즈
            cv2.resizeWindow("All Combined",
                            config.WINDOW_WIDTH * 3,
                            config.WINDOW_HEIGHT * 2 if DISPLAY_GRAPH else config.WINDOW_HEIGHT)
            cv2.imshow("All Combined", all_combined)
            
        if DISPLAY_OVERALL:
            graph.show_graph()

        #waitkey를 호출하지 않으면 카메라가 멈춤춤
        if cv2.waitKey(1) & 0xFF == 27:
            break
        
    cap.release()
    cv2.destroyAllWindows()
    gz.close_face_mesh()

class ConcentrationGraph:
    def __init__(self):
        self.max_frames = 100
        self.frame_number = 0
        self.scores = {
            'Overall': [],
            'Gaze': [],
            'Blink': [],
            'HeadPose': []
        }
        self.frame_numbers = []
        self._init_plot()

    def _init_plot(self):
        plt.style.use('dark_background')
        plt.ioff()
        self.fig, self.axs = plt.subplots(2, 2, figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        
        self.lines = {}
        for (i, (title, ax)) in enumerate(zip(self.scores.keys(), self.axs.flat)):
            ax.set_title(title, color='white')
            ax.set_xlim(0, self.max_frames)
            ax.set_ylim(0, 1.0)
            ax.grid(True, color='#707b7c', linestyle='--', alpha=0.7)
            ax.set_facecolor('black')
            for spine in ax.spines.values():
                spine.set_color('white')
            ax.tick_params(colors='white')
            line, = ax.plot([], [], color='gold', linewidth=2)
            self.lines[title] = line

    def _update_plot(self, overall, gaze, blink, headpose):
        # Add new data
        self.frame_numbers.append(self.frame_number)
        self.scores['Overall'].append(overall)
        self.scores['Gaze'].append(gaze)
        self.scores['Blink'].append(blink)
        self.scores['HeadPose'].append(headpose)
        self.frame_number += 1

        # Keep only max_frames data
        if len(self.frame_numbers) > self.max_frames:
            self.frame_numbers.pop(0)
            for key in self.scores:
                self.scores[key].pop(0)

        # Update each line
        for (title, line) in self.lines.items():
            line.set_xdata(self.frame_numbers)
            line.set_ydata(self.scores[title])

            ax = self._get_ax(title)
            if len(self.frame_numbers) > 1:
                ax.set_xlim(self.frame_numbers[0], self.frame_numbers[-1])

        self.fig.canvas.draw()


    def _get_ax(self, title):
        titles = list(self.scores.keys())
        idx = titles.index(title)
        return self.axs.flat[idx]


    def plot_to_image(self):
        self.canvas.draw()
        buffer = self.canvas.buffer_rgba()
        img_array = np.asarray(buffer)
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        return img_rgb

    def show_graph(self):
        plot_img = self.plot_to_image()
        plot_img_resized = cv2.resize(
            plot_img,
            (config.WINDOW_WIDTH_CONC, config.WINDOW_HEIGHT_CONC),
            interpolation=cv2.INTER_AREA
        )
        cv2.imshow("Concentration Tracking", plot_img_resized)



if __name__ == "__main__":
    main()