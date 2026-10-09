#!/bin/bash

# exclude: lidar point clouds (fast_lio cloud_*/Laser_map, raw livox/lidar), fast_lio accumulated path,
#          gazebo topics and gazebo servo_controller topics (sim only)
rosbag record -a -x "(.*)/cloud_(.*)|(.*)/Laser_map|(.*)/livox/lidar(.*)|(.*)/path$|/gazebo/(.*)|(.*)/servo_controller/(.*)" "$@"
