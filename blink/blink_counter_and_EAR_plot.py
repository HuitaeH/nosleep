import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from .FaceMeshModule import FaceMeshGenerator
from utils import DrawingUtils
import os


class BlinkCounterandEARPlot:
    """
    A class to detect and count eye blinks in a video using facial landmarks.
    
    This class processes video frames to detect faces, track eye movements,
    calculate Eye Aspect Ratio (EAR), plot EAR, and count blinks in real-time.
    """
    
    # Define facial landmark indices for eyes
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE_EAR = [33, 159, 158, 133, 153, 145]  # Points for EAR calculation
    LEFT_EYE_EAR = [362, 380, 374, 263, 386, 385]  # Points for EAR calculation
    
    # Define colors for visualization
    COLORS = {
        'GREEN': {'hex': '#56f10d', 'bgr': (86, 241, 13)},
        'BLUE': {'hex': '#0329fc', 'bgr': (30, 46, 209)},
        'RED': {'hex': '#f70202', 'bgr': None}
    }

    def __init__(self, video_path, threshold, consec_frames, save_video=False, output_filename=None):
        """
        Initialize the BlinkCounter with video and detection parameters.
        
        Args:
            video_path (str): Path to the input video file
            threshold (float): EAR threshold for blink detection
            consec_frames (int): Number of consecutive frames below threshold to count as a blink
            save_video (bool): Whether to save the processed video
            output_filename (str): Name of the output video file if saving
        """
        # Initialize core parameters
        self.generator = FaceMeshGenerator()
        self.video_path = video_path
        self.EAR_THRESHOLD = threshold
        self.CONSEC_FRAMES = consec_frames
        self.concentration_score = 100  # 초기 점수 100
        self.closed_eye_time = 0.0  # 눈 감은 시간 (초 단위)
        self.eye_open = True  # True면 눈 뜬 상태
        self.concentration_values = []  # ⭐️ 추가 (concentration 기록용 리스트)
        
        # Initialize video saving parameters
        self._init_video_saving(save_video, output_filename)
        
        # Initialize tracking variables
        self._init_tracking_variables()
        
        # Initialize plotting
        self._init_plot()

    def _init_video_saving(self, save_video, output_filename):
        """Initialize video saving parameters and create output directory if needed."""
        self.save_video = save_video
        self.output_filename = output_filename
        self.out = None
        
        if self.save_video and self.output_filename:
            save_dir = "DATA/VIDEOS/OUTPUTS"
            os.makedirs(save_dir, exist_ok=True)
            self.output_filename = os.path.join(save_dir, self.output_filename)

    def _init_tracking_variables(self):
        """Initialize variables used for tracking blinks and frame processing."""
        self.blink_counter = 0
        self.frame_counter = 0
        self.frame_number = 0
        self.ear_values = []
        self.frame_numbers = []
        self.max_frames = 100
        self.new_w = self.new_h = None
        # Add default y-axis limits
        self.default_ymin = 0.0  # Typical minimum EAR value
        self.default_ymax = 1.0  # Typical maximum EAR value

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
        self.ax.set_ylabel("EAR", color='white', fontsize=12)
        self.ax.set_title("Real-Time Eye Aspect Ratio (EAR)", 
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
        self.Y_vals = [self.EAR_THRESHOLD] * self.max_frames
        self.C_vals = [100] * self.max_frames  # ⭐️ 초기 concentration 100으로 설정

        # EAR curve
        self.EAR_curve, = self.ax.plot(
            self.x_vals,
            self.y_vals,
            color=self.COLORS['GREEN']['hex'],
            label="Eye Aspect Ratio",
            linewidth=2
        )

        # EAR Threshold line
        self.threshold_line, = self.ax.plot(
            self.x_vals,
            self.Y_vals,
            color=self.COLORS['RED']['hex'],
            label="Blink Threshold",
            linewidth=2,
            linestyle='--'
        )

        # Concentration curve ⭐️ 추가
        self.Curve_concentration, = self.ax.plot(
            self.x_vals,
            self.C_vals,
            color="#ffd700",  # Gold color
            label="Concentration Score",
            linewidth=2
        )

        # Legend 추가
        self.legend = self.ax.legend(
            handles=[self.EAR_curve, self.threshold_line, self.Curve_concentration],
            loc='upper right',
            fontsize=10,
            facecolor='black',
            edgecolor='white',
            labelcolor='white',
            framealpha=0.8,
            borderpad=1,
            handlelength=2
        )

    def eye_aspect_ratio(self, eye_landmarks, landmarks):
        """
        Calculate the eye aspect ratio (EAR) for given eye landmarks.
        
        The EAR is calculated using the formula:
        EAR = (||p2-p6|| + ||p3-p5||) / (2||p1-p4||)
        where p1-p6 are specific points around the eye.
        
        Args:
            eye_landmarks (list): Indices of landmarks for one eye
            landmarks (list): List of all facial landmarks
        
        Returns:
            float: Calculated eye aspect ratio
        """
        A = np.linalg.norm(np.array(landmarks[eye_landmarks[1]]) - 
                          np.array(landmarks[eye_landmarks[5]]))
        B = np.linalg.norm(np.array(landmarks[eye_landmarks[2]]) - 
                          np.array(landmarks[eye_landmarks[4]]))
        C = np.linalg.norm(np.array(landmarks[eye_landmarks[0]]) - 
                          np.array(landmarks[eye_landmarks[3]]))
        return (A + B) / (2.0 * C)

    def _update_plot(self, ear):
        if len(self.ear_values) > self.max_frames:
            self.ear_values.pop(0)
            self.frame_numbers.pop(0)
            self.concentration_values.pop(0)  # ⭐ concentration 값도 pop

        # Concentration 값을 0~1로 정규화해서 추가
        normalized_concentration = self.concentration_score / 100.0
        self.concentration_values.append(normalized_concentration)

        color = self.COLORS['BLUE']['hex'] if ear < self.EAR_THRESHOLD else self.COLORS['GREEN']['hex']

        self.EAR_curve.set_xdata(self.frame_numbers)
        self.EAR_curve.set_ydata(self.ear_values)
        self.EAR_curve.set_color(color)

        self.threshold_line.set_xdata(self.frame_numbers)
        self.threshold_line.set_ydata([self.EAR_THRESHOLD] * len(self.frame_numbers))

        self.Curve_concentration.set_xdata(self.frame_numbers)
        self.Curve_concentration.set_ydata(self.concentration_values)

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
                handles=[self.EAR_curve, self.threshold_line, self.Curve_concentration],
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
        self.ax.draw_artist(self.EAR_curve)
        self.ax.draw_artist(self.threshold_line)
        self.ax.draw_artist(self.Curve_concentration)
        self.ax.draw_artist(self.legend)
        self.fig.canvas.flush_events()



    def process_frame(self, frame):
        """
        Process a single frame to detect and analyze eyes.
        
        Returns:
            tuple: Processed frame and EAR value
        """
        frame, face_landmarks = self.generator.create_face_mesh(frame, draw=False)
        
        if not face_landmarks:
            return frame, None
            
        # Calculate EAR
        right_ear = self.eye_aspect_ratio(self.RIGHT_EYE_EAR, face_landmarks)
        left_ear = self.eye_aspect_ratio(self.LEFT_EYE_EAR, face_landmarks)
        ear = (right_ear + left_ear) / 2.0
        
        # Determine visualization color
        color = self.COLORS['BLUE']['bgr'] if ear < self.EAR_THRESHOLD else self.COLORS['GREEN']['bgr']
        
        # Draw landmarks and update blink counter
        self._draw_frame_elements(frame, face_landmarks, color)
        
        return frame, ear

    def _draw_frame_elements(self, frame, landmarks, color):
        """Draw eye landmarks and blink counter on the frame."""
        # Draw eye landmarks
        for eye in [self.RIGHT_EYE, self.LEFT_EYE]:
            for loc in eye:
                cv.circle(frame, (landmarks[loc]), 2, color, cv.FILLED)
        
        # Draw blink counter
        DrawingUtils.draw_text_with_bg(
            frame, f"Blinks: {self.blink_counter}", (0, 60),
            font_scale=2, thickness=3,
            bg_color=color, text_color=(0, 0, 0)
        )
        # Draw blink counter
        DrawingUtils.draw_text_with_bg(
            frame, f"Blinks: {self.blink_counter}", (0, 60),
            font_scale=2, thickness=3,
            bg_color=color, text_color=(0, 0, 0)
        )

        # Draw concentration score
        DrawingUtils.draw_text_with_bg(
            frame, f"Concentration: {int(self.concentration_score)}", (0, 120),
            font_scale=1.8, thickness=3,
            bg_color=(255, 215, 0),  # Gold color
            text_color=(0, 0, 0)
        )

    def _update_blink_detection(self, ear):
        self.ear_values.append(ear)
        self.frame_numbers.append(self.frame_number)
        
        # EAR 기반으로 눈 감/뜬 판단
        is_eye_closed = (ear < self.EAR_THRESHOLD)

        if is_eye_closed:
            self.frame_counter += 1
        else:
            if self.frame_counter >= self.CONSEC_FRAMES:
                self.blink_counter += 1
            self.frame_counter = 0

        # --- Concentration Score 업데이트 ---
        # 1프레임 시간 계산 (초 단위)
        fps = 30  # 기본 30fps 가정 (정확하게 하려면 cap.get으로 가져와도 됨)
        delta_time = 1 / fps

        if is_eye_closed:
            self.closed_eye_time += delta_time
            if self.closed_eye_time >= 5.0:
                # 눈 5초 이상 감고 있으면 점수 감소
                self.concentration_score = max(0, self.concentration_score - 0.5)
        else:
            self.closed_eye_time = 0.0
            # 눈 잘 뜨고 있으면 점수 회복
            self.concentration_score = min(100, self.concentration_score + 0.2)

        self.frame_number += 1

    def _update_visualization(self, frame, ear, fps):
        """Update the visualization including the plot and video output."""
        self._update_plot(ear)
        
        # Convert plot to image and resize
        plot_img = self.plot_to_image()
        plot_img_resized = cv.resize(
            plot_img,
            (frame.shape[1], int(plot_img.shape[0] * frame.shape[1] / plot_img.shape[1]))
        )
        
        # Stack frames and handle video output
        frame = cv.vconcat([frame, plot_img_resized])
        #self._handle_video_output(frame, fps)

    def _handle_video_output(self, stacked_frame, fps):
        """Handle video output, including saving and display."""
        # Initialize video writer if needed
        if self.new_w is None:
            self.new_w = stacked_frame.shape[1]
            self.new_h = stacked_frame.shape[0]
            if self.save_video:
                self.out = cv.VideoWriter(
                    self.output_filename,
                    cv.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (self.new_w, self.new_h)
                )

        # Save frame if requested
        if self.save_video:
            self.out.write(stacked_frame)

        # Display frame
        resizing_factor = 1.5
        resized_shape = (
            int(resizing_factor * stacked_frame.shape[1]),
            int(resizing_factor * stacked_frame.shape[0])
        )
        stacked_frame_resized = cv.resize(stacked_frame, resized_shape)
        cv.imshow("Video with EAR Plot", stacked_frame_resized)

    def plot_to_image(self):
        """Convert the matplotlib plot to an OpenCV-compatible image."""
        self.canvas.draw()
        
        buffer = self.canvas.buffer_rgba()
        img_array = np.asarray(buffer)
        
        # Convert RGBA to RGB
        img_rgb = cv.cvtColor(img_array, cv.COLOR_RGBA2RGB)
        return img_rgb


if __name__ == "__main__":
    # Example usage
    #input_video_path = "DATA/VIDEOS/INPUTS/blinking_1.mp4"
    input_video_path = 0
    blink_counter = BlinkCounterandEARPlot(
        video_path=input_video_path,
        threshold=0.294,
        consec_frames=3,
        save_video=True,
        output_filename="blinking_1_output.mp4"
    )
    blink_counter.process_video()
