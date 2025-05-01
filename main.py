
import cv2
import numpy as np
from headpose.main import HeadPose
from gaze.main import Gaze
from blink.main import Blink
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import config


DISPLAY_HEADPOSE = True
DISPLAY_GAZE = True
DISPLAY_BLINK = True

def main():
    print("Initializing modules...")
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    hp = HeadPose(display=DISPLAY_HEADPOSE, frame_width=frame_width, frame_height=frame_height)
    gz = Gaze(display=DISPLAY_GAZE)
    bk = Blink(display=DISPLAY_BLINK)
    graph = ConcentrationGraph()
    print("Modules initialized.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video frame.")
            break

        # ?�� 모듈 모두 0~1 ?��코어 반환
        head_score = hp.compute(frame)
        gaze_score = gz.compute(frame)
        blink_score = bk.compute(frame)

        # ?���? 집중?�� ?��?�� (�?�? ?���?)
        overall = (head_score*0.4 + gaze_score*0.4 + blink_score*0.2)
        graph._update_plot(overall)
        print(f"H: {head_score:.2f}, G: {gaze_score:.2f}, B: {blink_score:.2f}, overall: {overall:.2f}")
        graph.show_graph()
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    gz.close_face_mesh()

class ConcentrationGraph:
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
        self.concentration_values = []
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
        self.ax.set_ylabel("Concentration", color='white', fontsize=12)
        self.ax.set_title("Concentration Score", 
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
        self.C_vals = [100] * self.max_frames  # ⭐️ 초기 concentration 100?���? ?��?��

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

        # Legend 추�??
        self.legend = self.ax.legend(
            handles=[self.threshold_line, self.ConcentrationCurve],
            loc='upper right',
            fontsize=10,
            facecolor='black',
            edgecolor='white',
            labelcolor='white',
            framealpha=0.8,
            borderpad=1,
            handlelength=2
        )

    def _update_plot(self, value):
        if len(self.concentration_values) > self.max_frames:
            self.frame_numbers.pop(0)
            self.concentration_values.pop(0)

        # Concentration 값을 0~1�? ?��규화?��?�� 추�??
        normalized_concentration = value / 100.0
        self.concentration_values.append(normalized_concentration)
        self.frame_numbers.append(self.frame_number)
        self.frame_number += 1

        color = self.COLORS['BLUE']['hex'] if value < self.CONCENT_THRESHOLD else self.COLORS['GREEN']['hex']

        # self.EAR_curve.set_xdata(self.frame_numbers)
        # self.EAR_curve.set_ydata(self.contentration_value)
        # self.EAR_curve.set_color(color)

        self.threshold_line.set_xdata(self.frame_numbers)
        self.threshold_line.set_ydata([self.CONCENT_THRESHOLD] * len(self.frame_numbers))

        self.ConcentrationCurve.set_xdata(self.frame_numbers)
        self.ConcentrationCurve.set_ydata(self.concentration_values)

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
                handles=[self.threshold_line, self.ConcentrationCurve],
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
        self.ax.draw_artist(self.ConcentrationCurve)
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
        cv2.imshow("Overall Contenration Plot", plot_img_resized)




if __name__ == "__main__":
    main()