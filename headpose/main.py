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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import time
import numpy as np
import cv2
import config
FACE_DETECTOR = "./headpose/assets/face_detector.onnx"
FACE_LANDMARKS = "./headpose/assets/face_landmarks.onnx"
HEAD_DOWN_THRESHOLD = -10.0 # 10 이상 머리 떨굼 발생 시 스코어에 영향 

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
        self.pitch_up_start_time = None
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
        self.graph = HeadPoseGraph()
        self.frame = None
        
        pass

    
    def compute(self, frame: np.ndarray) -> float:

        print("HeadPose compute start")
        start_time = time.time()
        HeadPoseFrame = frame.copy()

        faces, _ = self.face_detector.detect(HeadPoseFrame, 0.7)
        pitch, yaw, roll = 0.0, 0.0, 0.0
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
                yaw = np.arctan2(-rotation_matrix[2, 0], sy)
                roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            else:
                pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                yaw = np.arctan2(-rotation_matrix[2, 0], sy)
                roll = 0

            pitch = np.degrees(pitch)
            yaw = np.degrees(yaw)
            roll = np.degrees(roll)
            self.graph._update_plot(pitch)

            print(f"Pitch: {pitch:.2f}, Yaw: {yaw:.2f}, Roll: {roll:.2f}")

            # pitch를 기반으로 점수 계산
            max_pitch = 30
            # score = max(0.0, 1.0 - abs(pitch) / max_pitch)

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
            cv2.putText(HeadPoseFrame, f"Yaw: {yaw:.1f}", (10, info_y_start + line_spacing),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(HeadPoseFrame, f"Roll: {roll:.1f}", (10, info_y_start + line_spacing * 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
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
            #cv2.namedWindow("HeadPose", cv2.WINDOW_NORMAL)
            #cv2.resizeWindow("HeadPose", config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
            cv2.putText(HeadPoseFrame, "HeadPose", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            #cv2.imshow("HeadPose", HeadPoseFrame)

            ## graph
            plot_img = self.graph.plot_to_image()
            plot_img_resized = cv2.resize(
                plot_img,
                (config.WINDOW_WIDTH, config.WINDOW_HEIGHT),
                interpolation=cv2.INTER_AREA
            )
            #cv2.imshow("HeadPose Plot", plot_img_resized)
            # 세로로 concat


            # (1) 타입(dtype) 맞추기: float → uint8
            if plot_img.dtype != np.uint8:
                # 0.0~1.0 범위라면 255 곱해주고, 클립 후 uint8 변환
                plot_img = np.clip(plot_img * 255, 0, 255).astype(np.uint8)

            # (2) 채널수 맞추기
            # - 그레이스케일 (ndim==2) → BGR
            if plot_img.ndim == 2:
                plot_img = cv2.cvtColor(plot_img, cv2.COLOR_GRAY2BGR)
            # - RGBA (4채널) → BGR
            elif plot_img.shape[2] == 4:
                # 만약 RGB순이면 COLOR_RGBA2BGR, BGR순이면 COLOR_BGRA2BGR
                plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)

            # (3) 컬러 순서 맞추기 (RGB → BGR)
            # Matplotlib 이미지는 보통 RGB이므로
            plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)

            # (4) 크기 맞추기: HeadPoseFrame 과 같은 너비/높이로
            h, w = HeadPoseFrame.shape[:2]
            plot_img_resized = cv2.resize(plot_img, (w, h), interpolation=cv2.INTER_AREA)

            self.frame = cv2.vconcat([HeadPoseFrame, plot_img_resized])
            # 또는 numpy로
            # combined = np.vstack((HeadPoseFrame, plot_img_resized))

            # # 하나의 창에 띄우기
            # cv2.namedWindow("HeadPose Combined", cv2.WINDOW_NORMAL)
            # # 창 크기도 세로가 두 배가 되도록 설정
            # cv2.resizeWindow("HeadPose Combined", config.WINDOW_WIDTH, config.WINDOW_HEIGHT * 2)
            # cv2.imshow("HeadPose Combined", self.frame)


        print("HeadPose compute end, time : ", time.time() - start_time)
        return self.score
class HeadPoseGraph:
    # Define colors for visualization
    COLORS = {
        'GREEN': {'hex': '#56f10d', 'bgr': (86, 241, 13)},
        'BLUE': {'hex': '#0329fc', 'bgr': (30, 46, 209)},
        'RED': {'hex': '#f70202', 'bgr': None}
    }

    def __init__(self):
        self.CONCENT_THRESHOLD = HEAD_DOWN_THRESHOLD  # Example threshold for EAR
        self._init_tracking_variables()
        self._init_plot()

    # def update(self, x, y):
    #     self.x_data.append(x)
    #     self.y_data.append(y)
    #     self.ax.clear()
    #     self.ax.plot(self.x_data, self.y_data, color='blue')
    #     plt.draw()
    def _init_tracking_variables(self):
        """Initialize variables used for tracking blinks and frame processing."""
        self.blink_counter = 0
        self.frame_counter = 0
        self.frame_number = 0
        #self.concentration_values = []
        self.frame_numbers = []
        self.h_angles = []
        self.max_frames = 100
        self.new_w = self.new_h = None
        # Add default y-axis limits
        self.default_ymin = -20.0  # Typical minimum EAR value
        self.default_ymax = +20.0  # Typical maximum EAR value

    def _init_plot(self):
        """Initialize the matplotlib plot for EAR visualization."""
        # Set up dark theme plot
        plt.style.use('dark_background')
        plt.ioff()
        self.fig, self.ax = plt.subplots(figsize=(8, 5), dpi=200)
        self.canvas = FigureCanvas(self.fig)
        
        # Configure plot aesthetics
        self._configure_plot_aesthetics()
        
        # Initialize plot data
        self._init_plot_data()

        self.fig.canvas.draw()

    def _configure_plot_aesthetics(self):
        """Configure the aesthetic properties of the plot."""
        # Set background colors
        self.fig.patch.set_facecolor('#000000')
        self.ax.set_facecolor('#000000')
        
        # Configure axes with default limits initially
        self.ax.set_ylim(self.default_ymin, self.default_ymax)
        self.ax.set_xlim(0, self.max_frames)
        
        # Set labels and title
        self.ax.set_xlabel("Frame Number", color='white', fontsize=12)
        self.ax.set_ylabel("HeadPose", color='white', fontsize=12)
        self.ax.set_title("HeadPose Degree Score", 
                         color='white', pad=10, fontsize=18, fontweight='bold')
        
        # Configure grid and spines
        self.ax.grid(True, color='#707b7c', linestyle='--', alpha=0.7)
        for spine in self.ax.spines.values():
            spine.set_color('white')
        
        # Configure ticks and legend
        self.ax.tick_params(colors='white', which='both')

    def _init_plot_data(self):
        self.x_vals = list(range(self.max_frames))
        self.y_vals = [0] * self.max_frames
        self.Y_vals = [self.CONCENT_THRESHOLD] * self.max_frames
        self.H_vals = [0] * self.max_frames

        # Threshold line
        self.threshold_line, = self.ax.plot(
            self.x_vals,
            self.Y_vals,
            color=self.COLORS['RED']['hex'],
            label="Blink Threshold",
            linewidth=2,
            linestyle='--'
        )

        self.Hcurve, = self.ax.plot(
            self.x_vals,
            self.H_vals,
            color= "#0329fc",
            label = "Horizontal Score",
            linewidth=2
        )


        # Legend 추가
        self.legend = self.ax.legend(
            handles=[self.threshold_line, self.Hcurve],
            loc='upper right',
            fontsize=10,
            facecolor='black',
            edgecolor='white',
            labelcolor='white',
            framealpha=0.8,
            borderpad=1,
            handlelength=2
        )

    def _update_plot(self, h_angle):
        if len(self.frame_numbers) > self.max_frames:
            self.frame_numbers.pop(0)
            self.h_angles.pop(0)
        
        # Concentration 값을 0~1로 정규화해서 추가
        # normalized_concentration = value / 100.0
        # self.concentration_values.append(normalized_concentration)
        self.frame_numbers.append(self.frame_number)
        self.frame_number += 1

        self.h_angles.append(h_angle)

        #color = self.COLORS['BLUE']['hex'] if value < self.CONCENT_THRESHOLD else self.COLORS['GREEN']['hex']

        # self.EAR_curve.set_xdata(self.frame_numbers)
        # self.EAR_curve.set_ydata(self.contentration_value)
        # self.EAR_curve.set_color(color)

        self.threshold_line.set_xdata(self.frame_numbers)
        self.threshold_line.set_ydata([self.CONCENT_THRESHOLD] * len(self.frame_numbers))

        # self.ConcentrationCurve.set_xdata(self.frame_numbers)
        # self.ConcentrationCurve.set_ydata(self.concentration_values)
        self.Hcurve.set_xdata(self.frame_numbers)
        self.Hcurve.set_ydata(self.h_angles)

        if len(self.frame_numbers) > 1:
            x_min = min(self.frame_numbers)
            x_max = max(self.frame_numbers)
            if x_min == x_max:
                x_min -= 0.5
                x_max += 0.5
            self.ax.set_xlim(x_min, x_max)
        else:
            self.ax.set_xlim(0, self.max_frames)

        if self.legend not in self.ax.get_children():
            self.legend = self.ax.legend(
                handles=[self.threshold_line, self.Hcurve],
                loc='upper right',
                fontsize=10,
                facecolor='black',
                edgecolor='white',
                labelcolor='white',
                framealpha=0.8,
                borderpad=1,
                handlelength=2
            )

        self.ax.draw_artist(self.ax.patch)
        self.ax.draw_artist(self.threshold_line)
        # self.ax.draw_artist(self.ConcentrationCurve)
        self.ax.draw_artist(self.Hcurve)
        self.ax.draw_artist(self.legend)
        self.fig.canvas.flush_events()
    
    def plot_to_image(self):
        """Convert the matplotlib plot to an OpenCV-compatible image."""
        self.canvas.draw()
        
        buffer = self.canvas.buffer_rgba()
        img_array = np.asarray(buffer)
        
        # Convert RGBA to RGB
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        return img_rgb

    def show_graph(self):
        plot_img = self.plot_to_image()
        plot_img_resized = cv2.resize(
            plot_img,
            (config.WINDOW_WIDTH_CONC, config.WINDOW_HEIGHT_CONC),
            interpolation=cv2.INTER_AREA
        )
        cv2.imshow("Gaze Plot", plot_img_resized)