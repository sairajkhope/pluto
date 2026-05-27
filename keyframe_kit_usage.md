# Robot Keyframe Kit - Quick Start Guide

A web-based keyframe editor for any MuJoCo robot.

**PyPI:** https://pypi.org/project/robot-keyframe-kit/

## Installation

```bash
pip install robot-keyframe-kit
```

## Basic Usage

### Command Line

```bash
# Launch the editor with your robot's MuJoCo XML
keyframe-editor path/to/robot.xml

# With custom options
keyframe-editor path/to/robot.xml --name my_robot --save-dir ./keyframes
```

Then open **http://localhost:8081** in your browser.

### Python API

```python
from robot_keyframe_kit import ViserKeyframeEditor, EditorConfig

# Minimal - just provide XML path
editor = ViserKeyframeEditor("path/to/robot.xml")

# With configuration
config = EditorConfig(
    name="my_robot",
    root_body="torso",
    save_dir="keyframes",
)
editor = ViserKeyframeEditor("path/to/robot.xml", config=config)

# Keep running
import time
while True:
    time.sleep(1.0)
```

## UI Overview

| Panel | Contents |
|-------|----------|
| **Left** | Save, keyframe list, sequence builder |
| **Center** | Left-side joint sliders |
| **Right** | Right-side joint sliders, mirror mode, settings |

### Key Features

- **Add Keyframe** - Save current pose as a keyframe
- **Update Keyframe** - Overwrite selected keyframe with current pose
- **Test Keyframe** - Play with physics simulation
- **Ground** - Place robot on the floor
- **Mirror Mode** - Auto-sync left/right joints
- **Save Motion** - Export to `.lz4` file

## Camera Controls

- **Scroll** - Zoom
- **Left-click + Drag** - Rotate
- **Right-click + Drag** - Pan

## Loading Saved Keyframes

```python
import joblib

data = joblib.load("keyframes/my_robot/motion.lz4")
print(data["keyframes"])        # List of keyframe dicts
print(data["timed_sequence"])   # [(name, time), ...]
print(data["qpos"])             # Recorded trajectory
```

## Example: ToddlerBot

```bash
# Clone the robot_keyframe_kit repo for example files
git clone https://github.com/Stanford-TML/robot_keyframe_kit.git
cd robot_keyframe_kit

# Run with example robot
keyframe-editor examples/toddlerbot_2xc/scene.xml --name toddlerbot
```

## Requirements

- Python ≥ 3.9
- MuJoCo ≥ 3.0
- Modern web browser

## Links

- **GitHub:** https://github.com/Stanford-TML/robot_keyframe_kit
- **PyPI:** https://pypi.org/project/robot-keyframe-kit/
- **Viser (visualization library):** https://github.com/nerfstudio-project/viser
