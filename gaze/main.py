
import numpy as np
import cv2
import config
import gaze.gaze_util as gaze_util
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import time
import mediapipe as mp


mp_face_mesh = mp.solutions.face_mesh  # initialize the face mesh model

##score 가중치
def get_gaze_score(vertical: float, horizontal: float, v_off, h_off) -> float:
    """
    Calculate the gaze score based on vertical and horizontal angles.
    This is a placeholder function. You can implement your own logic here.
    """
    # Example: simple average of vertical and horizontal angles as a score
    ## Vertical
    vertical = vertical - v_off
    horizontal = horizontal - h_off
    v_score = get_score(vertical, 5, 10)
    h_score = get_score(horizontal, 15, 20)

    ### horizontal
    return v_score * h_score  # Adjust weights as needed

def get_score(value: float, threshold: float, max_range: float) -> float:
    abs_val = abs(value)
    if abs_val <= threshold:
        return 1.0
    elif abs_val >= max_range:
        return 0.0
    else:
        return np.interp(abs_val, [threshold, max_range], [1.0, 0.0])

class Gaze:
    def __init__(self, display: bool = False):
        self.display = display
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.graph = GazeGraph()  # Concentration graph instance
        self.horizontal = 0.0
        self.vertical = 0.0
        self.frame = None

        ## for calibration
        self.button = True
        self.num_frame = 0
        self.FRAME_THRESHOLD = 50
        self.gaze_horizontal_off = 0.0
        self.gaze_vertical_off = 0.0
        self.gaze_horizontal_sum = 0.0
        self.gaze_vertical_sum = 0.0

        ##for error handling
        self.prev_score = 0.0

    def close_face_mesh(self):

        #호출필요

        self.face_mesh.close()

    def compute(self, frame: np.ndarray) -> float:

        print("Gaze compute start")
        start_time = time.time()
        GazeFrame = frame.copy()

        GazeFrame.flags.writeable = False
        GazeFrame = cv2.cvtColor(GazeFrame, cv2.COLOR_BGR2RGB)  # frame to RGB for the face-mesh model
        results = self.face_mesh.process(GazeFrame)
        GazeFrame = cv2.cvtColor(GazeFrame, cv2.COLOR_RGB2BGR)  # frame back to BGR for OpenCV

        if results.multi_face_landmarks:
            vertical, horizontal = gaze_util.gaze(GazeFrame, results.multi_face_landmarks[0])  # gaze estimation
            self.vertical = vertical
            self.horizontal = horizontal
            self.graph._update_plot(self.vertical, self.horizontal)  # Update the graph with the new angles


            # calibration
            self.num_frame += 1
            if self.button:
                self.gaze_horizontal_sum += horizontal
                self.gaze_vertical_sum += vertical

                if self.num_frame == self.FRAME_THRESHOLD:
                    self.gaze_horizontal_off = self.gaze_horizontal_sum / self.FRAME_THRESHOLD
                    self.gaze_vertical_off = self.gaze_vertical_sum / self.FRAME_THRESHOLD
                    self.button = False

        
        if self.display:
            # Display the frame with gaze overlay (if needed)
            
            #cv2.namedWindow("Gaze", cv2.WINDOW_NORMAL)
            #cv2.resizeWindow("Gaze", config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
            cv2.putText(GazeFrame, "Gaze", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            cv2.putText(GazeFrame,
                    f"H: {self.horizontal:.1f}°, V: {self.vertical:.1f}°",
                    (100, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0), 2)
            
            cv2.putText(GazeFrame, f"offset : h: {self.gaze_horizontal_off:.1f} v: {self.gaze_vertical_off:.1f}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            #cv2.imshow("Gaze", GazeFrame)

            ## graph
            plot_img = self.graph.plot_to_image()
            plot_img_resized = cv2.resize(
                plot_img,
                (config.WINDOW_WIDTH, config.WINDOW_HEIGHT),
                interpolation=cv2.INTER_AREA
            )
            #cv2.imshow("Gaze Plot", plot_img_resized)

            # 3) dtype → uint8
            if plot_img.dtype != np.uint8:
                plot_img = np.clip(plot_img * 255, 0, 255).astype(np.uint8)

            # 4) 채널 수 맞추기: gray→BGR, RGBA→BGR
            if plot_img.ndim == 2:
                plot_img = cv2.cvtColor(plot_img, cv2.COLOR_GRAY2BGR)
            elif plot_img.shape[2] == 4:
                plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)

            # 5) 색순서 맞추기: RGB→BGR
            plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)

            # 6) 동일한 크기로 리사이즈
            h, w = GazeFrame.shape[:2]
            plot_img_resized = cv2.resize(
                plot_img,
                (w, h),
                interpolation=cv2.INTER_AREA
            )

            # 7) 두 이미지를 세로로 이어붙이기
            self.frame = cv2.vconcat([GazeFrame, plot_img_resized])
            # cv2.namedWindow("Gaze Combined", cv2.WINDOW_NORMAL)
            # # 8) 창 크기 조정 및 출력
            # cv2.resizeWindow("Gaze Combined",  config.WINDOW_WIDTH, config.WINDOW_HEIGHT * 2)
            # cv2.imshow("Gaze Combined", self.frame)
        print("Gaze compute end, elapsed time : ", time.time() - start_time)
        if results.multi_face_landmarks:
            # Calculate the gaze score based on the vertical and horizontal angles
            # Here we can use a simple average of the angles as a score, or any other logic
            window_size = 50
            recent_v = self.graph.v_angles[-window_size:]
            recent_h = self.graph.h_angles[-window_size:]

            scores = [
                get_gaze_score(v, h, self.gaze_vertical_off, self.gaze_horizontal_off)
                for v, h in zip(recent_v, recent_h)
            ]

            # 평균 집중도 계산
            concentration_score = sum(scores) / len(scores) if scores else self.prev_score
            self.prev_score = concentration_score
            
            return concentration_score
        return 0.0
    
class GazeGraph:
    # Define colors for visualization
    COLORS = {
        'GREEN': {'hex': '#56f10d', 'bgr': (86, 241, 13)},
        'BLUE': {'hex': '#0329fc', 'bgr': (30, 46, 209)},
        'RED': {'hex': '#f70202', 'bgr': None}
    }

    def __init__(self):
        self.CONCENT_THRESHOLD = 0.294  # Example threshold for EAR
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
        self.v_angles = []
        self.h_angles = []
        self.frame_numbers = []
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
        self.ax.set_ylabel("Eye degree", color='white', fontsize=12)
        self.ax.set_title("Gaze Degree Score", 
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

        self.C_vals = [100] * self.max_frames  # ⭐️ 초기 concentration 100으로 설정
        self.V_vals = [0] * self.max_frames
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


        # Concentration curve ⭐️ 추�??
        self.ConcentrationCurve, = self.ax.plot(
            self.x_vals,
            self.C_vals,
            color="#ffd700",  # Gold color
            label="Concentration Score",
            linewidth=2
        )



        self.Vcurve, = self.ax.plot(
            self.x_vals,
            self.V_vals,
            color= "#56f10d",
            label = "Vertical Score",
            linewidth=2
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
            handles=[self.threshold_line, self.ConcentrationCurve, self.Vcurve, self.Hcurve],
            loc='upper right',
            fontsize=10,
            facecolor='black',
            edgecolor='white',
            labelcolor='white',
            framealpha=0.8,
            borderpad=1,
            handlelength=2
        )

    def _update_plot(self, v_angle, h_angle):
        if len(self.frame_numbers) > self.max_frames:
            self.frame_numbers.pop(0)

            self.v_angles.pop(0)
            self.h_angles.pop(0)
        
        # Concentration 값을 0~1로 정규화해서 추가
        # normalized_concentration = value / 100.0
        # self.concentration_values.append(normalized_concentration)

        self.frame_numbers.append(self.frame_number)
        self.frame_number += 1

        self.v_angles.append(v_angle)
        self.h_angles.append(h_angle)

        #color = self.COLORS['BLUE']['hex'] if value < self.CONCENT_THRESHOLD else self.COLORS['GREEN']['hex']

        # self.EAR_curve.set_xdata(self.frame_numbers)
        # self.EAR_curve.set_ydata(self.contentration_value)
        # self.EAR_curve.set_color(color)

        self.threshold_line.set_xdata(self.frame_numbers)
        self.threshold_line.set_ydata([self.CONCENT_THRESHOLD] * len(self.frame_numbers))

        # self.ConcentrationCurve.set_xdata(self.frame_numbers)
        # self.ConcentrationCurve.set_ydata(self.concentration_values)
        self.Vcurve.set_xdata(self.frame_numbers)
        self.Vcurve.set_ydata(self.v_angles)
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
                handles=[self.threshold_line, self.Vcurve, self.Hcurve],
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
        self.ax.draw_artist(self.Vcurve)
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