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
        pass

    def compute(self, frame: np.ndarray) -> float:

        print("HeadPose compute start")
        start_time = time.time()
        HeadPoseFrame = frame.copy()
        # Step 1: Get faces from current frame.
        faces, _ = self.face_detector.detect(HeadPoseFrame, 0.7)

        # Any valid face found?
        if len(faces) > 0:
            self.tm.start()

            # Step 2: Detect landmarks. Crop and feed the face area into the
            # mark detector. Note only the first face will be used for
            # demonstration.
            face = refine(faces, self.frame_width, self.frame_height, 0.15)[0]
            x1, y1, x2, y2 = face[:4].astype(int)
            patch = frame[y1:y2, x1:x2]

            # Run the mark detection.
            marks = self.mark_detector.detect([patch])[0].reshape([68, 2])

            # Convert the locations from local face area to the global image.
            marks *= (x2 - x1)
            marks[:, 0] += x1
            marks[:, 1] += y1

            # Step 3: Try pose estimation with 68 points.
            pose = self.pose_estimator.solve(marks)

            self.tm.stop()

            # All done. The best way to show the result would be drawing the
            # pose on the frame in realtime.

            # Do you want to see the pose annotation?
            self.pose_estimator.visualize(HeadPoseFrame, pose, color=(0, 255, 0))

            # Do you want to see the axes?
            # pose_estimator.draw_axes(frame, pose)

            # Do you want to see the marks?
            # mark_detector.visualize(frame, marks, color=(0, 255, 0))

            # Do you want to see the face bounding boxes?
            # face_detector.visualize(frame, faces)



        if self.display:
            # Display the frame with head pose overlay (if needed)
            # Draw the FPS on screen.
            cv2.rectangle(HeadPoseFrame, (0, 0), (90, 30), (0, 0, 0), cv2.FILLED)
            cv2.putText(HeadPoseFrame, f"FPS: {self.tm.getFPS():.0f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            
            cv2.namedWindow("HeadPose", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("HeadPose", config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
            cv2.putText(HeadPoseFrame, "HeadPose", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            cv2.imshow("HeadPose", HeadPoseFrame)
            pass
        print("HeadPose compute end, time : ", time.time() - start_time)
        return 0.0