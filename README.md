# OpenMind Interaction Autonomy

OpenMind-based AI-robot interaction and autonomy integration framework

## Overview

**OpenMind Interaction Autonomy is a framework that integrates the OpenMind platform with ROS2-based autonomous navigation systems**, enabling seamless communication between multimodal AI agents and physical robots. The framework combines OpenMind's AI capabilities with KIST's autonomous navigation system to create a unified robotics platform.

## Capabilities

### OpenMind Platform
* **Modular Architecture**: Designed with Python for simplicity and seamless integration
* **Multimodal AI Agents**: Process diverse inputs like web data, camera feeds, and LIDAR
* **LLM/VLM Integration**: Support for multiple LLMs and Visual Language Models
* **Hardware Support via Plugins**: Supports new hardware through plugins for API endpoints and specific robot hardware connections to `ROS2`, `Zenoh`, and `CycloneDDS`
* **Web-Based Debugging Display**: Monitor the system in action with WebSim (available at http://localhost:8000/) for easy visual debugging

### Vendor Autonomous Navigation
* **RealSense Camera Integration**: Depth and RGB stream processing with depth-color alignment
* **GPU-Accelerated Processing**: CUDA-based PointCloud generation and BEV Occupancy Grid
* **Navigation Planning**: GPS-based Global Planner and DWA Local Planner for obstacle avoidance
* **Robot Control**: Unitree Go2 quadruped robot control bridge

## Architecture Overview

**Goal**: Fully integrate vendor autonomous navigation code into OpenMind platform as native modules.

The framework integrates vendor ROS2 autonomous navigation code directly into the OpenMind platform:

```
┌─────────────────────────────────────────────────────────┐
│              OpenMind Platform                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │  AI Agent Runtime (LLM/VLM, Decision Making)     │  │
│  └──────────────────┬───────────────────────────────┘  │
│                     │                                    │
│  ┌──────────────────▼───────────────────────────────┐  │
│  │  Integrated Vendor Modules                       │  │
│  │  • RealSense Camera (Input)                      │  │
│  │  • PointCloud Generation (Processing)             │  │
│  │  • BEV Occupancy Grid (Processing)               │  │
│  │  • GPS/DWA Navigation (Action)                   │  │
│  │  • Unitree Go2 Control (Action)                  │  │
│  └──────────────────────────────────────────────────┘  │
│                     │                                    │
│  ┌──────────────────▼───────────────────────────────┐  │
│  │  ROS2 Integration Layer                          │  │
│  │  (DDS/Zenoh middleware)                          │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

Vendor code is integrated as OpenMind platform modules (inputs/actions), not as a separate external system.

## Getting Started

To get started with OpenMind Interaction Autonomy, you'll need to set up both the OpenMind platform and the ROS2-based vendor system.

### Prerequisites

You will need the [`uv` package manager](https://docs.astral.sh/uv/getting-started/installation/) for Python package management.

### Clone the Repository

```bash
git clone <repository-url>
cd openmind-interaction-autonomy
```

### Install System Dependencies

#### For Linux
```bash
sudo apt-get update
sudo apt-get install -y \
    portaudio19-dev \
    python3-dev \
    ffmpeg \
    libasound2-dev \
    libv4l-dev \
    build-essential \
    cmake \
    python3-pip \
    python3-colcon-common-extensions
```

#### For macOS
```bash
brew install portaudio ffmpeg
```

### Set Up OpenMind Platform

1. **Create virtual environment**
   ```bash
   uv venv
   source .venv/bin/activate  # On Linux/macOS
   ```

2. **Install Python dependencies**
   ```bash
   uv pip install -e .
   # For DDS support
   uv pip install -e ".[dds]"
   ```

3. **Obtain an OpenMind API Key**
   - Get your API Key at [OpenMind Portal](https://portal.openmind.org/)
   - Copy it to `config/<agent-name>.json5`, replacing the `openmind_free` placeholder
   - Or, create a `.env` file and add: `OM_API_KEY=your_api_key`

4. **Launch OpenMind Platform**
   ```bash
   uv run src/run.py <agent-name>
   ```

### Set Up Vendor ROS2 System

1. **Build ROS2 workspace**
   ```bash
   cd vendor/interaction_autonomous_navigation/Autonomous_Navigation
   colcon build
   source install/setup.bash
   ```

2. **Launch RealSense camera**
   ```bash
   ros2 launch realsense2_camera rs_launch.py \
       depth_module.depth_profile:=640x480x30 \
       rgb_camera.color_profile:=640x480x30 \
       enable_depth:=true \
       enable_color:=true \
       pointcloud.enable:=false \
       align_depth.enable:=true
   ```

3. **Run PointCloud generation**
   ```bash
   ros2 run pointcloud_xyzrgb pointcloud_gpu_node
   ```

4. **Run DWA Planner**
   ```bash
   ros2 run dwa_nav dwa_node
   ```

After launching both systems, they will communicate via DDS/Zenoh middleware.

**Note**: Make sure ROS2 is installed and sourced before running ROS2 commands:
```bash
source /opt/ros/<rosdistro>/setup.bash
```

## Dependencies

### Hardware Requirements

#### Minimum Requirements
- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5 or equivalent)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 20GB free space
- **Network**: Ethernet or WiFi connection

#### For OpenMind Platform
- **Camera**: USB webcam or compatible camera (for vision inputs)
- **Audio**: Microphone and speakers (for ASR/TTS)
- **Display**: Monitor for WebSim debugging interface

#### For Vendor Autonomous Navigation
- **GPU**: NVIDIA GPU with CUDA support (required for PointCloud and BEV processing)
  - NVIDIA Jetson AGX Orin / Thor (recommended)
  - NVIDIA GTX 1060 or better (desktop)
- **Camera**: Intel RealSense D435/D455 depth camera
- **Robot**: Unitree Go2 quadruped robot (for full autonomy)
- **GPS**: RTK-GPS module (for GPS-based navigation)
- **LIDAR**: RPLiDAR (optional, for SLAM)

### Software Requirements

#### Operating System
- **Linux**: Ubuntu 22.04 (recommended) or Ubuntu 20.04
- **macOS**: macOS 12.0+ (limited support)
- **Windows**: Not officially supported

#### Core Software
- **Python**: >= 3.10
- **ROS2**: Humble Hawksbill or Iron Irwini
  - Required for vendor autonomous navigation
- **CUDA**: 11.0+ (for GPU acceleration)
- **CMake**: >= 3.5 (for ROS2 package building)

#### Python Dependencies
Managed via `pyproject.toml`:
- Core: `numpy`, `opencv-python`, `pillow`, `json5`
- AI/ML: `torch`, `torchvision`, `tensorflow>=2.15.0`, `ultralytics`
- Communication: `eclipse-zenoh>=1.4.0`, `websockets`, `fastapi`
- Audio: `pyaudio`, `soundfile`, `sounddevice`
- Hardware: `pyserial`, `bleak`, `pynmeagps`
- Optional: `cyclonedds==0.10.2` (for DDS communication)

#### ROS2 Dependencies
Vendor packages require:
- `rclpy`, `rclcpp` (ROS2 Python/C++ clients)
- `geometry_msgs`, `sensor_msgs`, `nav_msgs`
- `realsense2_camera` (RealSense ROS2 driver)
- `unitree_go` (Unitree Go2 messages)
- CUDA-enabled packages for GPU processing

## What's Next?

* Try out different agent configurations in the `/config/` directory
* Explore vendor ROS2 packages and customize navigation parameters
* Add new `inputs` and `actions` to extend functionality
* Design custom agents by creating your own `json5` config files
* Change system prompts in configuration files to create new behaviors
* Integrate additional hardware sensors and actuators

## Interfacing with Robot Hardware

The framework assumes that robot hardware provides a high-level SDK that accepts elemental movement and action commands. OpenMind can interface with your hardware abstraction layer (HAL) via:

* **ROS2**: Standard ROS2 topics and services
* **Zenoh**: Recommended for all new development
* **CycloneDDS**: DDS middleware for distributed systems
* **Websockets**: For web-based interfaces
* **USB/Serial**: Direct hardware connections

If your robot hardware does not yet provide a suitable HAL, you may need to create one using traditional robotics approaches, simulation environments (Unity, Gazebo), and custom VLAs.

## Recommended Development Platforms

The framework is developed and tested on:

* **Nvidia Thor** (running JetPack 7.0) - full support
* **Jetson AGX Orin 64GB** (running Ubuntu 22.04 and JetPack 6.1) - limited support
* **Mac Studio** with Apple M2 Ultra with 48 GB unified memory (running macOS Sequoia)
* **Mac Mini** with Apple M4 Pro with 48 GB unified memory (running macOS Sequoia)
* **Generic Linux machines** (running Ubuntu 22.04)

The framework _should_ run on other platforms (such as Windows) and microcontrollers such as the Raspberry Pi 5 16GB.

## Project Structure

```
.
├── src/              # OpenMind platform source
├── vendor/           # ROS2 autonomous navigation system
│   └── interaction_autonomous_navigation/
│       ├── Autonomous_Navigation/  # ROS2 workspace
│       └── LLMagent/              # LLM agent
├── config/           # Configuration files
├── docs/             # Documentation
└── pyproject.toml    # Python project configuration
```

## Detailed Documentation

More detailed documentation can be accessed at:
* [OpenMind Documentation](https://docs.openmind.org/)
* [ROS2 Documentation](https://docs.ros.org/)

## Contributing

Please make sure to read the [Contributing Guide](./CONTRIBUTING.md) before making a pull request.

## License

This project is licensed under the terms of the MIT License, which is a permissive free software license that allows users to freely use, modify, and distribute the software.
