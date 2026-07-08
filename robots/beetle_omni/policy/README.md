## Beetle Omni RL Controller

### Launch

Temporary USB symbolic link
```
sudo ln -s /dev/ttyUSB0 /dev/flight_controller
```

#### Configuration

Check `config/BeetleRLAgentReal.yaml` and `config/BeetleRLAgentSim.yaml` to set parameters before launching.

Key flags:
```yaml
ideal_obs: false            # must be false on real hardware
gimbal_effort_ctrl: false   # must be false on real hardware
enable_gimbal: true         # enable/disable gimbal control (useful for debugging)
enable_thrust: true         # enable/disable rotor thrust (useful for debugging)
```

---

#### Step 1: Check Joint Default Positions (Real Hardware)

Set `enable_gimbal: false` and `enable_thrust: false`, then launch the controller:
```bash
roslaunch beetle_omni bringup_rl_omni.launch real_machine:=True simulation:=False headless:=True
```
Set `enable_save: false` and `gimbal_range: 0.0` in `launch/response.launch`, then launch:
```bash
roslaunch beetle_omni response.launch
```
Verify that all joints are in their default positions.

---

#### Step 2: Check Joint Control (Real Hardware)

Set `enable_thrust: false` and `enable_gimbal: true`, then launch the controller:
```bash
roslaunch beetle_omni bringup_rl_omni.launch real_machine:=True simulation:=False headless:=True
```
Hold the platform and rotate/shake the robot to check whether joint control is stable.
> **Warning:** If oscillations are observed, do **not** take off.

Check rotor feedback stability 
```shell
rosrun beetle_omni ESCTelemUpdateAnalysis.py _duration:=20
# or run until Ctrl+C:
python3 ESCTelemUpdateAnalysis.py _duration:=0
```
---

#### Step 3: Launch Controller

Simulation:
```bash
roslaunch beetle_omni bringup_rl_omni.launch real_machine:=False simulation:=True headless:=False
```
Real-world:
```bash
roslaunch beetle_omni bringup_rl_omni.launch real_machine:=True simulation:=False headless:=True
```

Launch the keyboard interface, press **`r`** to arm and **`t`** to take off:
```bash
rosrun aerial_robot_base keyboard_command.py
```

---

#### Goal / Trajectory Interface

```bash
roscd beetle_omni
```

1. Interactive 3D goal in RViz:
    ```bash
    python scripts/rl/interactive_marker.py
    ```
2. Publish a fixed pose target (`x y z Roll Pitch Yaw` in m / degrees):
    ```bash
    python scripts/rl/pub_desired_pose.py 0.0 0.0 1.0 0 0 0
    ```
3. Publish a sequence of waypoints:
    ```bash
    python scripts/rl/pub_waypoints.py
    ```
4. Publish a trajectory:
    ```bash
    python scripts/rl/pub_trajectory.py
    ```

---

#### Runtime: Reload Policy

Hot-swap the ONNX policy without restarting the controller (replace `beetle_omni` with your robot namespace):
```bash
rosservice call /beetle_omni/rlagent/reload_policy \
  "policy_name: 'policy/policy-2026-01-09_20-11-41-20000.onnx'"
```

---

#### Runtime: Update Parameters

Update any parameter via `rosparam set`, then call `reload_params` to apply:
```bash
rosparam set /beetle_omni/rlagent/enable_thrust true
rosparam set /beetle_omni/rlagent/enable_gimbal true
rosparam set /beetle_omni/rlagent/thrust_scale 1.0
rosparam set /beetle_omni/rlagent/goal_pos_z 0.8

rosservice call /beetle_omni/rlagent/reload_params {}
```

Reloadable parameters: `enable_thrust`, `enable_gimbal`, `thrust_scale`, `thrust_default`, `thrust_limit`, `thrust_tau`, `gimbal_kp`, `gimbal_kd`, `goal_pos_x/y/z`, `goal_angle_R/P/Y`, `rlagent_debug`, `rlagent_forward_info`, `rlagent_verbose`.

---

### Foxglove Data Visualization

#### Installation

Install from apt:
```bash
sudo apt install ros-$ROS_DISTRO-foxglove-bridge
```

Or build from source:
```bash
sudo apt install libwebsocketpp-dev    # WebSocket support
sudo apt install nlohmann-json3-dev    # JSON for Modern C++ (header-only)
sudo apt install libasio-dev           # Async I/O library
catkin config --cmake-args -DCATKIN_ENABLE_TESTING=OFF  # disable test builds
git clone https://github.com/foxglove/ros-foxglove-bridge.git
git clone https://github.com/StefanFabian/ros_babel_fish.git
git clone https://github.com/RobotWebTools/rosbridge_suite.git -b ros1
rosdep update
rosdep install --ignore-src --default-yes --from-path src
catkin build foxglove_bridge
```

#### Usage

```bash
# Web dashboard: https://app.foxglove.dev/dragon-lab/dashboard
roslaunch --screen foxglove_bridge foxglove_bridge.launch port:=8765
```

---

### Motor Test

Calibrate the thrust/torque → PWM mapping. See [motor test README](/aerial_robot_nerve/motor_test/README.md).
