# BALLTRAC

A fast, reliable tennis ball tracking system built with Python and OpenCV. Supports real-time tracking in live webcam input and video files, with automatic or manual color calibration.

## Features

- Real-time ball detection using HSV color filtering.
- Dual operation modes:
  - **Real mode**: Tracks a physical tennis ball using a webcam.
  - **Screen mode**: Tracks a ball in digital videos or simulations on a screen.
- Automatic and manual HSV calibration options.
- UI overlay displays current mode, calibration status, and FPS.
- Option to toggle visibility of the calibration region.

## Demo
<div style="display: flex; justify-content: space-between;">
  <img src="imgs/img1.jpg" alt="img1" style="width: 45%;"/>
  <img src="imgs/img2.jpg" alt="img2" style="width: 45%;"/>
</div>

## Installation

### Requirements

- Python 3.7 or higher
- OpenCV (`opencv-python`)
- NumPy
- imutils

### Installation Steps

```bash
pip install opencv-python numpy imutils
```

## Usage

### Run in real mode (default)

```bash
python balltrac.py
```

### Run in screen mode

```bash
python balltrac.py -s
```

### Enable manual HSV calibration

```bash
python balltrac.py -c
```

## Controls

| Key | Function                         |
| --- | -------------------------------- |
| q   | Quit the program                 |
| s   | Toggle between real/screen mode  |
| a   | Auto-calibrate using green box   |
| b   | Show or hide the calibration box |

## How It Works

- Converts incoming frames to HSV color space.
- Applies a mask based on HSV range to isolate the tennis ball.
- Uses erosion and dilation to reduce noise in the mask.
- Detects the largest contour and fits a circle around it.
- In auto-calibration, HSV values are sampled from the calibration region.
- In manual mode, HSV values are controlled via sliders.

## Tips

- Ensure the ball is fully within the green box before auto-calibration.
- Use stable, even lighting for consistent tracking performance.
- Manual calibration is recommended for difficult lighting environments.

