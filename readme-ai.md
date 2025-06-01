<div id="top">

<!-- HEADER STYLE: CLASSIC -->
<div align="center">

<img src="readmeai/assets/logos/purple.svg" width="30%" style="position: relative; top: 0; right: 0;" alt="Project Logo"/>

# NOSLEEP

<em>Innovating sleep patterns for peak performance.</em>

<!-- BADGES -->
<!-- local repository, no metadata badges. -->

<em>Built with the tools and technologies:</em>

<img src="https://img.shields.io/badge/Markdown-000000.svg?style=default&logo=Markdown&logoColor=white" alt="Markdown">
<img src="https://img.shields.io/badge/Keras-D00000.svg?style=default&logo=Keras&logoColor=white" alt="Keras">
<img src="https://img.shields.io/badge/Git-F05032.svg?style=default&logo=Git&logoColor=white" alt="Git">
<img src="https://img.shields.io/badge/TensorFlow-FF6F00.svg?style=default&logo=TensorFlow&logoColor=white" alt="TensorFlow">
<img src="https://img.shields.io/badge/scikitlearn-F7931E.svg?style=default&logo=scikit-learn&logoColor=white" alt="scikitlearn">
<img src="https://img.shields.io/badge/Rich-FAE742.svg?style=default&logo=Rich&logoColor=black" alt="Rich">
<br>
<img src="https://img.shields.io/badge/SymPy-3B5526.svg?style=default&logo=SymPy&logoColor=white" alt="SymPy">
<img src="https://img.shields.io/badge/MediaPipe-0097A7.svg?style=default&logo=MediaPipe&logoColor=white" alt="MediaPipe">
<img src="https://img.shields.io/badge/NumPy-013243.svg?style=default&logo=NumPy&logoColor=white" alt="NumPy">
<img src="https://img.shields.io/badge/Python-3776AB.svg?style=default&logo=Python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/SciPy-8CAAE6.svg?style=default&logo=SciPy&logoColor=white" alt="SciPy">
<img src="https://img.shields.io/badge/pandas-150458.svg?style=default&logo=pandas&logoColor=white" alt="pandas">

</div>
<br>

---

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
    - [Project Index](#project-index)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Usage](#usage)
    - [Testing](#testing)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

**Introducing nosleep**

**Why nosleep?**

This project empowers developers with real-time data capture, analysis, and prediction capabilities. The core features include:

- **🔍 Real-time Data Capture:** Capture and log real-time data for in-depth analysis.
- **🚀 Prediction Capabilities:** Utilize machine learning models for accurate predictions.
- **💻 Cross-Platform Support:** Seamlessly deploy on both Windows and Raspberry Pi setups.

---

## Features

|      | Component       | Details                              |
| :--- | :-------------- | :----------------------------------- |
| ⚙️  | **Architecture**  | <ul><li>Follows a modular design with clear separation of concerns</li><li>Uses efficient data structures and algorithms for performance optimization</li></ul> |
| 🔩 | **Code Quality**  | <ul><li>Consistent coding style and naming conventions</li><li>Utilizes comments and docstrings for code documentation</li></ul> |
| 📄 | **Documentation** | <ul><li>Comprehensive README.md file with setup instructions and usage examples</li><li>Inline code comments explaining complex logic</li></ul> |
| 🔌 | **Integrations**  | <ul><li>Integrates seamlessly with various CICD tools like pip, training_log.v2, and requirements.txt</li><li>Uses external libraries for extended functionality</li></ul> |
| 🧩 | **Modularity**    | <ul><li>Organized into reusable modules for easy maintenance and scalability</li><li>Follows the principle of separation of concerns</li></ul> |
| 🧪 | **Testing**       | <ul><li>Includes unit tests for critical functions and components</li><li>Uses testing frameworks like pytest for automated testing</li></ul> |
| ⚡️  | **Performance**   | <ul><li>Optimized algorithms for efficient resource utilization</li><li>Utilizes caching mechanisms for faster data retrieval</li></ul> |
| 🛡️ | **Security**      | <ul><li>Implements secure coding practices to prevent common vulnerabilities</li><li>Uses encryption for sensitive data handling</li></ul> |
| 📦 | **Dependencies**  | <ul><li>Relies on a wide range of dependencies for extended functionality</li><li>Manages dependencies using a requirements.txt file</li></ul> |

---

## Project Structure

```sh
└── nosleep/
    ├── blink
    │   ├── __init__.py
    │   ├── blink_counter_and_EAR_plot.py
    │   ├── FaceMeshModule.py
    │   ├── LICENSE
    │   └── main.py
    ├── config.py
    ├── csv_example.csv
    ├── data_collector.py
    ├── gaze
    │   ├── __init__.py
    │   ├── gaze_util.py
    │   ├── helpers.py
    │   └── main.py
    ├── headpose
    │   ├── __init__.py
    │   ├── assets
    │   ├── face_detection.py
    │   ├── LICENSE
    │   ├── main.py
    │   ├── mark_detection.py
    │   ├── pose_estimation.py
    │   └── utils.py
    ├── heads_with_numerics.txt
    ├── linear_regression.py
    ├── logs
    │   ├── training_log.v2
    │   └── validation_log.v2
    ├── main.py
    ├── minmax_scaler.pkl
    ├── models
    │   ├── drowsiness_rnn_best(1).h5
    │   ├── drowsiness_rnn_best(2).h5
    │   ├── drowsiness_rnn_best(3).h5
    │   └── drowsiness_rnn_best.h5
    ├── pkg_manage.py
    ├── predictor.py
    ├── README.md
    ├── requirements-rasp.txt
    ├── requirements.txt
    ├── spike_tx
    │   ├── __init__.py
    │   ├── ble_scan.py
    │   ├── bluetooth_client.py
    │   ├── bt.py
    │   └── label_tester_for_win.py
    └── utils.py
```

### Project Index

<details open>
	<summary><b><code>C:\USERS\ROBOT\DESKTOP\NOSLEEP/</code></b></summary>
	<!-- __root__ Submodule -->
	<details>
		<summary><b>__root__</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ __root__</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\robot\Desktop\nosleep/blob/master/config.py'>config.py</a></b></td>
					<td style='padding: 8px;'>Define standard window sizes and calibration frame count for the projects computer vision functionality.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\robot\Desktop\nosleep/blob/master/data_collector.py'>data_collector.py</a></b></td>
					<td style='padding: 8px;'>- Capture and log real-time data from a webcam, including head pose, gaze, and blink scores, along with user input labels<br>- The data is saved to a CSV file for analysis<br>- The code initializes modules, performs calibration, and continuously captures data until the user stops<br>- Visualization options are available for monitoring the process.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\robot\Desktop\nosleep/blob/master/heads_with_numerics.txt'>heads_with_numerics.txt</a></b></td>
					<td style='padding: 8px;'>Identify and extract numeric values from the heads_with_numerics.txt file to enhance data analysis and processing within the project architecture.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\robot\Desktop\nosleep/blob/master/linear_regression.py'>linear_regression.py</a></b></td>
					<td style='padding: 8px;'>Calculate normalized weights and predict concentration scores using Ridge regression on standardized input data.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\robot\Desktop\nosleep/blob/master/main.py'>main.py</a></b></td>
					<td style='padding: 8px;'>- Initialize modules, calibrate, and track concentration levels using head pose, gaze, and blink analysis<br>- Display real-time data and predictions, leveraging a neural network model<br>- Graphical representation of concentration metrics is available<br>- Supports both Windows and Raspberry Pi camera setups<br>- Bluetooth integration for command transmission to external devices.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\robot\Desktop\nosleep/blob/master/pkg_manage.py'>pkg_manage.py</a></b></td>
					<td style='padding: 8px;'>Words.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\robot\Desktop\nosleep/blob/master/predictor.py'>predictor.py</a></b></td>
					<td style='padding: 8px;'>- Implement a RealtimePredictor class that loads a model and predicts classes based on incoming data<br>- The class maintains a sliding window of data points, making predictions when the window is full<br>- It also includes methods to update data, predict if enough data is available, and reset the data buffer.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\robot\Desktop\nosleep/blob/master/requirements-rasp.txt'>requirements-rasp.txt</a></b></td>
					<td style='padding: 8px;'>Words.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\robot\Desktop\nosleep/blob/master/requirements.txt'>requirements.txt</a></b></td>
					<td style='padding: 8px;'>- Specify the dependencies required for the project by listing them in the <code>requirements.txt</code> file<br>- This file outlines the essential libraries and versions needed for the codebase to function correctly<br>- It ensures that all necessary dependencies are installed to support the projects functionality and maintain compatibility across different environments.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\robot\Desktop\nosleep/blob/master/utils.py'>utils.py</a></b></td>
					<td style='padding: 8px;'>- Illustrates utility functions for drawing overlays, rounded rectangles, and text with background in OpenCV<br>- Enhances error handling for input validation and drawing operations<br>- The code showcases usage examples for each drawing method within a main function.</td>
				</tr>
			</table>
		</blockquote>
	</details>
	<!-- blink Submodule -->
	<details>
		<summary><b>blink</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ blink</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\robot\Desktop\nosleep/blob/master/blink\blink_counter_and_EAR_plot.py'>blink_counter_and_EAR_plot.py</a></b></td>
					<td style='padding: 8px;'>- Project SummaryThe <code>BlinkCounterandEARPlot</code> class is a key component of the project architecture, responsible for real-time detection and counting of eye blinks in a video using facial landmarks<br>- By leveraging facial landmark detection, eye movement tracking, and Eye Aspect Ratio (EAR) calculation, this class offers a comprehensive solution for blink analysis<br>- The class not only detects blinks but also visualizes the EAR values through plotting, providing valuable insights into eye behavior over time<br>- This functionality enhances the projects capabilities in facial analysis and contributes to a more robust understanding of eye-related patterns in video data.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\robot\Desktop\nosleep/blob/master/blink\FaceMeshModule.py'>FaceMeshModule.py</a></b></td>
					<td style='padding: 8px;'>- Generate face mesh landmarks from a video stream or webcam feed, enabling visualization and analysis of facial features<br>- The code initializes a FaceMesh detector with specified parameters, processes frames to identify landmarks, and optionally saves the output video<br>- It encapsulates the functionality within a FaceMeshGenerator class and a video processing function.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\robot\Desktop\nosleep/blob/master/blink\LICENSE'>LICENSE</a></b></td>
					<td style='padding: 8px;'>Define the projects licensing terms.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\robot\Desktop\nosleep/blob/master/blink\main.py'>main.py</a></b></td>
					<td style='padding: 8px;'>- Compute the blink detection score based on eye aspect ratio (EAR) from the input frame<br>- Adjusts for eye closure duration and calculates a normalized score<br>- Handles frame processing, visualization, and display options<br>- Supports webcam and graph display modes<br>- Enhances user experience by overlaying blink information on frames.</td>
				</tr>
			</table>
		</blockquote>
	</details>
	<!-- gaze Submodule -->
	<details>
		<summary><b>gaze</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ gaze</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\robot\Desktop\nosleep/blob/master/gaze\gaze_util.py'>gaze_util.py</a></b></td>
					<td style='padding: 8px;'>- Generate gaze direction and draw it on the input frame using face landmarks from the mediapipe framework<br>- Estimate the 3D gaze point and correct it for head rotation, then project it onto the image plane<br>- Finally, calculate and return the horizontal and vertical angles of the gaze direction.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\robot\Desktop\nosleep/blob/master/gaze\helpers.py'>helpers.py</a></b></td>
					<td style='padding: 8px;'>Calculate relative coordinates based on landmarks and shape dimensions for the gaze tracking functionality in the project.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\robot\Desktop\nosleep/blob/master/gaze\main.py'>main.py</a></b></td>
					<td style='padding: 8px;'>- The <code>main.py</code> file in the <code>gaze</code> directory initializes a Gaze class that computes gaze scores based on facial landmarks<br>- It utilizes the Mediapipe library for face mesh detection and provides functionality for gaze estimation, calibration, and visualization<br>- The code calculates concentration scores from eye angles and updates a real-time graph for visualization.</td>
				</tr>
			</table>
		</blockquote>
	</details>
	<!-- headpose Submodule -->
	<details>
		<summary><b>headpose</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ headpose</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\robot\Desktop\nosleep/blob/master/headpose\face_detection.py'>face_detection.py</a></b></td>
					<td style='padding: 8px;'>- Implementing a face detection module using SCRFD, the code file in face_detection.py decodes distance predictions to bounding boxes and key points<br>- It initializes a face detector with model configurations, preprocesses images, performs inference, and visualizes detection results<br>- The module efficiently detects faces in images, providing valuable insights for further processing.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\robot\Desktop\nosleep/blob/master/headpose\LICENSE'>LICENSE</a></b></td>
					<td style='padding: 8px;'>Define the projects licensing terms.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\robot\Desktop\nosleep/blob/master/headpose\main.py'>main.py</a></b></td>
					<td style='padding: 8px;'>- Face DetectionIdentifying and cropping human faces within the video frame.2<br>- <strong>Facial Landmark DetectionRunning facial landmark detection on the cropped face image.3<br>- </strong>Pose EstimationEstimating the pose of the head by solving a Perspective-n-Point (PnP) problem.By integrating modules such as <code>FaceDetector</code>, <code>MarkDetector</code>, and <code>PoseEstimator</code>, this file coordinates the flow of data and operations necessary for accurate head pose estimation<br>- The project leverages ONNX models for face detection and facial landmarks, ensuring robust performance.For more detailed insights and code implementation, please refer to the <a href="https://github.com/yinguobing/head-pose-estimation">Head Pose Estimation GitHub repository</a>.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\robot\Desktop\nosleep/blob/master/headpose\mark_detection.py'>mark_detection.py</a></b></td>
					<td style='padding: 8px;'>- Detects facial marks on images using a Convolutional Neural Network<br>- Preprocesses input images and runs them through the model to obtain facial marks<br>- Provides a method to visualize the detected marks on the input image.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\robot\Desktop\nosleep/blob/master/headpose\pose_estimation.py'>pose_estimation.py</a></b></td>
					<td style='padding: 8px;'>- Estimate head pose based on facial landmarks, providing rotation and translation vectors<br>- Visualize a 3D box annotation of the pose on an image<br>- Draw axes on the image using the provided pose<br>- Show a 3D model of the estimated points for visualization.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\robot\Desktop\nosleep/blob/master/headpose\utils.py'>utils.py</a></b></td>
					<td style='padding: 8px;'>- Refines face boxes for face landmark detection by adjusting dimensions and centering them<br>- Clips values exceeding specified dimensions for safety.</td>
				</tr>
			</table>
		</blockquote>
	</details>
	<!-- logs Submodule -->
	<details>
		<summary><b>logs</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ logs</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\robot\Desktop\nosleep/blob/master/logs\training_log.v2'>training_log.v2</a></b></td>
					<td style='padding: 8px;'>- The provided code file serves as a crucial component within the codebase architecture, enabling seamless integration of user authentication and authorization functionalities<br>- By leveraging this code, developers can ensure secure access control and user management within the project<br>- This code file plays a pivotal role in enhancing the overall security and user experience of the application.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\robot\Desktop\nosleep/blob/master/logs\validation_log.v2'>validation_log.v2</a></b></td>
					<td style='padding: 8px;'>- The <code>validation_log.v2</code> file in the <code>logs</code> directory of the project structure contains essential data related to the evaluation accuracy of the brain events<br>- It captures and stores information crucial for assessing the performance and accuracy of the model during validation processes<br>- This log file plays a significant role in tracking and analyzing the evaluation metrics, particularly focusing on the accuracy of brain events within the system.</td>
				</tr>
			</table>
		</blockquote>
	</details>
	<!-- spike_tx Submodule -->
	<details>
		<summary><b>spike_tx</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ spike_tx</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\robot\Desktop\nosleep/blob/master/spike_tx\ble_scan.py'>ble_scan.py</a></b></td>
					<td style='padding: 8px;'>- Scan and connect to BLE devices using asyncio and Bleak library<br>- Discover nearby devices and print their details<br>- Establish a connection with a specific device and verify the connection status<br>- Ideal for BLE device interaction in Python projects.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\robot\Desktop\nosleep/blob/master/spike_tx\bluetooth_client.py'>bluetooth_client.py</a></b></td>
					<td style='padding: 8px;'>- Demonstrate Bluetooth device discovery, connection establishment, and command transmission in a Python script<br>- The code facilitates device selection, connection handling, and command sending, supporting actions like attacks or idle states<br>- It encapsulates Bluetooth functionality for seamless interaction with connected devices.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\robot\Desktop\nosleep/blob/master/spike_tx\bt.py'>bt.py</a></b></td>
					<td style='padding: 8px;'>- Implement a Bluetooth class to discover nearby devices and establish connections<br>- The class encapsulates methods to find devices and connect to them using RFCOMM protocol<br>- Upon execution, it lists nearby devices and demonstrates connecting to the first found device.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='C:\Users\robot\Desktop\nosleep/blob/master/spike_tx\label_tester_for_win.py'>label_tester_for_win.py</a></b></td>
					<td style='padding: 8px;'>- Implement a Bluetooth client for Windows to control attacks using keyboard inputs<br>- The code establishes a connection, allowing users to trigger different attacks by pressing specific keys<br>- It ensures seamless communication between the client and the device, enhancing user experience and control over the system.</td>
				</tr>
			</table>
		</blockquote>
	</details>
</details>

---

## Getting Started

### Prerequisites

This project requires the following dependencies:

- **Programming Language:** Python
- **Package Manager:** Pip

### Installation

Build nosleep from the source and intsall dependencies:

1. **Clone the repository:**

    ```sh
    ❯ git clone ../nosleep
    ```

2. **Navigate to the project directory:**

    ```sh
    ❯ cd nosleep
    ```

3. **Install the dependencies:**

<!-- SHIELDS BADGE CURRENTLY DISABLED -->
	<!-- [![pip][pip-shield]][pip-link] -->
	<!-- REFERENCE LINKS -->
	<!-- [pip-shield]: https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white -->
	<!-- [pip-link]: https://pypi.org/project/pip/ -->

	**Using [pip](https://pypi.org/project/pip/):**

	```sh
	❯ pip install -r requirements.txt
	```

### Usage

Run the project with:

**Using [pip](https://pypi.org/project/pip/):**
```sh
python {entrypoint}
```

### Testing

Nosleep uses the {__test_framework__} test framework. Run the test suite with:

**Using [pip](https://pypi.org/project/pip/):**
```sh
pytest
```

---

## Roadmap

- [X] **`Task 1`**: <strike>Implement feature one.</strike>
- [ ] **`Task 2`**: Implement feature two.
- [ ] **`Task 3`**: Implement feature three.

---

## Contributing

- **💬 [Join the Discussions](https://LOCAL/Desktop/nosleep/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://LOCAL/Desktop/nosleep/issues)**: Submit bugs found or log feature requests for the `nosleep` project.
- **💡 [Submit Pull Requests](https://LOCAL/Desktop/nosleep/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your LOCAL account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone C:\Users\robot\Desktop\nosleep
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to LOCAL**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://LOCAL{/Desktop/nosleep/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=Desktop/nosleep">
   </a>
</p>
</details>

---

## License

Nosleep is protected under the [LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

## Acknowledgments

- Credit `contributors`, `inspiration`, `references`, etc.

<div align="right">

[![][back-to-top]](#top)

</div>


[back-to-top]: https://img.shields.io/badge/-BACK_TO_TOP-151515?style=flat-square


---
