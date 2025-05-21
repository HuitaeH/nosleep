
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

DISPLAY = True              # camera display
DISPLAY_GRAPH = False       # graph display
DISPLAY_OVERALL = False     # overall concentration display

def main():
    print("Initializing modules...")
    cap = cv2.VideoCapture(0)
    # resolution settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    hp = HeadPose(display=DISPLAY, display_graph = DISPLAY_GRAPH, frame_width=640, frame_height=480)
    gz = Gaze(display=DISPLAY, display_graph = DISPLAY_GRAPH)
    bk = Blink(display=DISPLAY, display_graph = DISPLAY_GRAPH)
    graph = ConcentrationGraph()
    print("Modules initialized.")

        # 2. 캘리브레이션 시작
    frame_id = 0
    print("Start calibration. press 'c' to start.")
    while True:
        if keyboard.is_pressed('c'):
            print("Calibration started.")
            break
    while True:
        if (not hp.button and not gz.button) :
            break
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        head_score = hp.compute(frame)
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
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video frame.")
            break

        # ?�� 모듈 모두 0~1 ?��코어 반환
        head_score, pitch = hp.compute(frame)
        bk.blink_counter.set_pitch(pitch)
        gaze_score = gz.compute(frame)
        blink_score = bk.compute(frame)


        # 전체 집중도 예시 (가중 평균)
        overall = (head_score*0.3 + gaze_score*0.3 + blink_score*0.4)
        if (DISPLAY_OVERALL):
            graph._update_plot(overall, gaze_score, blink_score, head_score)
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