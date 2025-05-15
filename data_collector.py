# data_collector.py
import cv2
import numpy as np
import csv
import os
import keyboard  # pip install keyboard
from datetime import datetime

import config
from headpose.main import HeadPose
from gaze.main import Gaze
from blink.main import Blink
from main import ConcentrationGraph

DISPLAY = True
DISPLAY_OVERALL = False
DISPLAY_GRAPH = False

def get_label_from_key():
    if keyboard.is_pressed('1'):
        return 1
    elif keyboard.is_pressed('2'):
        return 2
    else:
        return 0

def main():
    print("Initializing modules...")
    # 1. 타임스탬프 기반 파일 이름 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("datasets", exist_ok=True)
    filename = f"datasets/{timestamp}.csv"

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    hp = HeadPose(display=DISPLAY, display_graph = DISPLAY_GRAPH, frame_width=640, frame_height=480)
    gz = Gaze(display=DISPLAY, display_graph = DISPLAY_GRAPH)
    bk = Blink(display=DISPLAY, display_graph = DISPLAY_GRAPH)
    graph = ConcentrationGraph()

    calib_done = False
    print("Modules initialized.")

    # 2. 캘리브레이션 시작
    frame_id = 0
    print("Start calibration. prees 'c' to start.")
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

    frame_id = 0

    # 파일 존재 여부 확인
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["frame", "headpose", "gaze", "blink", "label"])

        print("Start capturing data. Press ESC to stop.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera.")
                break

            head_score = hp.compute(frame)
            gaze_score = gz.compute(frame)
            blink_score = bk.compute(frame)

            label = get_label_from_key()
            writer.writerow([
                frame_id,
                round(head_score, 6),
                round(gaze_score, 6),
                round(blink_score, 6),
                label
            ])
            print(f"[{frame_id}] H: {head_score:.2f}, G: {gaze_score:.2f}, B: {blink_score:.2f}, Label: {label}")

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
                cv2.imshow("All Combined", all_combined)
                
            if DISPLAY_OVERALL:
                overall = (head_score*0.3 + gaze_score*0.3 + blink_score*0.4)
                graph._update_plot(overall, gaze_score, blink_score, head_score)
                graph.show_graph()
                #cv2.imshow("Frame", frame)
            cv2.waitKey(1)
            if keyboard.is_pressed('esc'):
                break

    cap.release()
    cv2.destroyAllWindows()
    gz.close_face_mesh()
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    main()
