

### Foxglove Data Visualization

#### Installation
Install from apt
```bash
sudo apt install ros-$ROS_DISTRO-foxglove-bridge
```
or build fron source code
```bash
sudo apt install libwebsocketpp-dev # Websocket
sudo apt install nlohmann-json3-dev # JSON for Modern C++, header-only C++ library
sudo apt install libasio-dev        # IO library
catkin config --cmake-args -DCATKIN_ENABLE_TESTING=OFF # ban sone test build
git clone https://github.com/foxglove/ros-foxglove-bridge.git
git clone https://github.com/StefanFabian/ros_babel_fish.git
git clone https://github.com/RobotWebTools/rosbridge_suite.git -b ros1
# Workspace
rosdep update
rosdep install --ignore-src --default-yes --from-path src
catkin build foxglove_bridge
```

#### Usage
```bash
#  Web View: https://app.foxglove.dev/dragon-lab/dashboard
roslaunch --screen foxglove_bridge foxglove_bridge.launch port:=8765
```


#### Motor Test
Calibrate the thrust/torque => PWM cmd, please refer [motor test](/aerial_robot_nerve/motor_test/README.md)