import cv2
from headpose.main import HeadPose
from gaze.main import Gaze
from blink.main import Blink


DISPLAY_HEADPOSE = True
DISPLAY_GAZE = True
DISPLAY_BLINK = True

def main():
    print("Initializing modules...")
    cap = cv2.VideoCapture(0)
    hp = HeadPose(display=DISPLAY_HEADPOSE)
    gz = Gaze(display=DISPLAY_GAZE)
    bk = Blink(display=DISPLAY_BLINK)
    print("Modules initialized.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 세 모듈 모두 0~1 스코어 반환
        head_score = hp.compute(frame)
        gaze_score = gz.compute(frame)
        blink_score = bk.compute(frame)

        # 전체 집중도 예시 (가중 평균)
        overall = (head_score*0.4 + gaze_score*0.4 + blink_score*0.2)
        print(f"H: {head_score:.2f}, G: {gaze_score:.2f}, B: {blink_score:.2f} → O: {overall:.2f}")

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()