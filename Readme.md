---

# Illegal Parking Detection System

This repository contains a computer vision-based system for detecting illegal parking in video streams. The system utilizes the YOLOv4 object detection model to identify cars within the video frames and checks whether they are parked in designated parking spots. If a violation is detected, it highlights the offending vehicle for further action.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Illegal parking is a common problem in urban areas, leading to traffic congestion and safety hazards. Traditional methods of enforcement can be labor-intensive and inefficient. This project aims to automate the detection of illegal parking using computer vision techniques, providing a more efficient solution for urban management.

The system employs the YOLOv4 (You Only Look Once version 4) object detection model, trained on the COCO dataset, to recognize vehicles within a video stream. By defining designated parking areas and comparing the detected vehicle locations against these predefined zones, the system identifies instances of illegal parking. Detected violations are highlighted in the output video for further review and enforcement.

## Requirements

To run the illegal parking detection system, you need the following dependencies:

- Python 3.x
- OpenCV
- NumPy
- imutils

These dependencies can be installed using pip with the provided `requirements.txt` file.

## Installation

Follow these steps to set up the system:

1. Clone the repository:

```bash
git clone https://github.com/prakharninja0927/illegal-parking-detection.git
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To use the system:

1. Place the input video files in the `adminResources/input/` directory.
2. Run the `illegal_parking_detection.py` script:

```bash
python illegal_parking_detection.py
```

3. The output video with detected violations will be saved in the `adminResources/output/` directory.

## Configuration

The behavior of the system can be customized through various configuration parameters:

- `UPLOAD_INPUT_CAMERA`: Path to the input video file.
- `UPLOAD_OUTPUT_CAMERA`: Directory to save the output video.
- `default_confidence`: Minimum confidence threshold for detecting objects.
- `default_threshold`: Threshold for non-maximum suppression.
- Paths to YOLOv4 model weights and configuration file.
- Path to the file containing parking coordinates (`parking-coordinate/test1.txt`).

Adjust these parameters according to your specific requirements.

## Contributing

Contributions to the project are welcome! If you have any suggestions, feature requests, or bug reports, please create a pull request or open an issue on GitHub.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to expand upon this README with additional sections, such as project architecture, examples, or performance metrics. This detailed README provides users with comprehensive information on the project's purpose, usage, and customization options.
