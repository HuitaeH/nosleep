"""
There are three major steps:
1. Detect and crop the human faces in the video frame.
2. Run facial landmark detection on the face image.
3. Estimate the pose by solving a PnP problem.

For more details, please refer to:
https://github.com/yinguobing/head-pose-estimation
"""

from .face_detection import FaceDetector
from .mark_detection import MarkDetector
from .pose_estimation import PoseEstimator
from .utils import refine

import time
import numpy as np
import cv2
import config
FACE_DETECTOR = "./headpose/assets/face_detector.onnx"
FACE_LANDMARKS = "./headpose/assets/face_landmarks.onnx"
class HeadPose:
    def __init__(self, display: bool = False, frame_width: int = 640, frame_height: int = 480):
        self.display = display
        # video_src = 0
        # cap = cv2.VideoCapture(video_src)
        # print(f"Video source: {video_src}")

        # Get the frame size. This will be used by the following detectors.
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Setup a face detector to detect human faces.
        self.face_detector = FaceDetector(FACE_DETECTOR)

        # Setup a mark detector to detect landmarks.
        self.mark_detector = MarkDetector(FACE_LANDMARKS)

        # Setup a pose estimator to solve pose.
        self.pose_estimator = PoseEstimator(frame_width, frame_height)

        # Measure the performance with a tick meter.
        self.tm = cv2.TickMeter()

        # Head down condition management 
        self.is_pitch_down = False
        self.pitch_down_start_time = None
        self.pitch_down_duration = 0.0 # cumulative time 
        self.PITCH_THRESHOLD = -10.0  # (예: pitch가 -20도 이하로 내려가면 고개 숙임으로 간주)
        self.is_pitch_up = True
        self.pitch_up_start_time = 0.0
        self.pitch_up_duration = 0.0
        
        # scoring 
        self.score = 1.0
        self.HEAD_DOWN_THRESHOLD = 2.0 # 2초 이상 머리 떨굼 발생 시 스코어에 영향 
        self.DECAY_RATE = 0.1 # 점수 감소 rate
        self.RECOVERY_RATE = 0.05 # 점수 회복 rate
        self.prev_time = time.time()
        self.HEAD_UP_THRESHOLD = 2.0 # 2초 이상 고개 들고 있을 시 점수 회복 시작 

        # plotting
        self.score_history = []

        # 250508 button pressed
        self.button = True
        self.num_frame = 0
        self.FRAME_THRESHOLD = 50
        self.default_pitch = 0.0
        pass

    
    def compute(self, frame: np.ndarray) -> float:
        print("HeadPose compute")
        HeadPoseFrame = frame.copy()

        faces, _ = self.face_detector.detect(HeadPoseFrame, 0.7)
        pitch = 0.0
        default_pitch = self.default_pitch
        
        # pitch, yaw, roll = 0.0, 0.0, 0.0
        if len(faces) > 0:
            self.tm.start()

            face = refine(faces, self.frame_width, self.frame_height, 0.15)[0]
            x1, y1, x2, y2 = face[:4].astype(int)
            patch = frame[y1:y2, x1:x2]

            marks = self.mark_detector.detect([patch])[0].reshape([68, 2])
            marks *= (x2 - x1)
            marks[:, 0] += x1
            marks[:, 1] += y1

            pose = self.pose_estimator.solve(marks)

            self.tm.stop()

            self.pose_estimator.visualize(HeadPoseFrame, pose, color=(0, 255, 0))

            # 여기 수정
            rotation_vector, translation_vector = pose

            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
            singular = sy < 1e-6

            if not singular:
                pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                # yaw = np.arctan2(-rotation_matrix[2, 0], sy)
                # roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            else:
                pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                # yaw = np.arctan2(-rotation_matrix[2, 0], sy)
                # roll = 0
            
            self.num_frame += 1
            if self.button == True and self.num_frame <= self.FRAME_THRESHOLD:
                pitch = np.degrees(pitch)
                default_pitch += pitch 
                self.default_pitch = default_pitch
                print(f"Frame Number: {self.num_frame}")
                print(f"Default Pitch(cumulative): {default_pitch:.2f}")
                print(f"Pitch(measuring default): {pitch:.2f}")

                if self.num_frame == self.FRAME_THRESHOLD:
                    default_pitch = default_pitch/self.num_frame
                    self.default_pitch = default_pitch
                    self.button = False
                    print(f"Frame Number: {self.num_frame}")
                    print(f"Default Pitch(Final): {default_pitch:.2f}")
                    print(f"Pitch(mesuring default): {pitch:.2f}")

            
                # yaw = np.degrees(yaw)
                # roll = np.degrees(roll)
                
                # print(f"Pitch: {pitch:.2f}, Yaw: {yaw:.2f}, Roll: {roll:.2f}")

                # pitch를 기반으로 점수 계산
               
                # score = max(0.0, 1.0 - abs(pitch) / max_pitch)
            else:  
                max_pitch = 30
                print(f"Default Pitch(fixed): {default_pitch:.2f}")
                pitch = np.degrees(pitch) - default_pitch
                print(f"Pitch: {pitch:.2f}")
                current_time = time.time()  # 현재 시간(초) 
                if pitch <= self.PITCH_THRESHOLD:
                    if not self.is_pitch_down:
                        # 새로 고개를 숙이기 시작한 경우
                        self.is_pitch_down = True
                        self.pitch_down_start_time = current_time
                        self.pitch_up_start_time = 0.0
                        self.is_pitch_up = False
                        self.pitch_up_duration = 0.0
                    else:
                        # 이미 숙이고 있는 상태 → 지속 시간 계산
                        self.pitch_down_duration = current_time - self.pitch_down_start_time
                else:
                    if self.is_pitch_down:
                        # 고개를 다시 들었을 때 
                        self.is_pitch_down = False
                        self.is_pitch_up = True 
                        self.pitch_down_start_time = None
                        self.pitch_up_start_time = current_time
                        self.pitch_down_duration = 0.0
                    else: 
                        # 이미 들고 있는 상태 -> 지속 시간 계산
                        self.pitch_up_duration = current_time - self.pitch_up_start_time

                if self.pitch_down_duration <= self.HEAD_DOWN_THRESHOLD or self.is_pitch_up == True:
                    recovery_rate = self.RECOVERY_RATE
                    concentration_time = max(0.0, self.pitch_up_duration - self.HEAD_UP_THRESHOLD)
                    self.score = min(1.0, self.score + recovery_rate * concentration_time) 
                else:
                    # 고개 숙인 시간이 길수록 score 감소
                    decay_rate = self.DECAY_RATE  # 초당 0.1씩 감소
                    excess_time = max(0.0, self.pitch_down_duration - self.HEAD_DOWN_THRESHOLD)
                    self.score = max(0.0, self.score - decay_rate * excess_time)

                # 매 프레임 score 저장
                self.score_history.append(self.score)

        if self.display:
            # FPS 표시
            cv2.rectangle(HeadPoseFrame, (0, 0), (90, 30), (0, 0, 0), cv2.FILLED)
            cv2.putText(HeadPoseFrame, f"FPS: {self.tm.getFPS():.0f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            
            # pitch, yaw, roll 표시
            info_y_start = 40  # 시작 y좌표 (FPS 바로 아래)
            line_spacing = 20  # 각 줄 간격

            cv2.putText(HeadPoseFrame, f"Pitch: {pitch:.1f}", (10, info_y_start),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            # cv2.putText(HeadPoseFrame, f"Yaw: {yaw:.1f}", (10, info_y_start + line_spacing),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            # cv2.putText(HeadPoseFrame, f"Roll: {roll:.1f}", (10, info_y_start + line_spacing * 2),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            # pitch down 지속 시간
            cv2.putText(HeadPoseFrame, f"Down Time: {self.pitch_down_duration:.1f}s",
                        (10, info_y_start + line_spacing * 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(HeadPoseFrame, f"UP Time: {self.pitch_up_duration:.1f}s",
                        (10, info_y_start + line_spacing * 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # 현재 score
            cv2.putText(HeadPoseFrame, f"Score: {self.score:.2f}",
                        (10, info_y_start + line_spacing * 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

            # 창 설정
            cv2.namedWindow("HeadPose", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("HeadPose", config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
            cv2.putText(HeadPoseFrame, "HeadPose", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imshow("HeadPose", HeadPoseFrame)


        return self.score

