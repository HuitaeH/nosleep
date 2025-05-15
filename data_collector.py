# data_collector.py
import cv2
import numpy as np
import csv
import os
import keyboard  # pip install keyboard
from datetime import datetime

from headpose.main import HeadPose
from gaze.main import Gaze
from blink.main import Blink

DISPLAY = False

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

    hp = HeadPose(display=DISPLAY, frame_width=640, frame_height=480)
    gz = Gaze(display=DISPLAY)
    bk = Blink(display=DISPLAY)
    print("Modules initialized.")

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
                cv2.imshow("Frame", frame)

            if keyboard.is_pressed('esc'):
                break

    cap.release()
    cv2.destroyAllWindows()
    gz.close_face_mesh()
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    main()
