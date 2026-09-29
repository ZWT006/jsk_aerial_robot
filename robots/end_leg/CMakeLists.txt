cmake_minimum_required(VERSION 3.0.2)
project(end_leg)

add_compile_options(-std=c++17 -Wall -Wextra)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS roscpp sensor_msgs std_msgs
)

include_directories(${catkin_INCLUDE_DIRS})

add_executable(leg_servo_calibration_node src/leg_servo_calibration.cpp)
add_dependencies(leg_servo_calibration_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(leg_servo_calibration_node ${catkin_LIBRARIES})

install(TARGETS leg_servo_calibration_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch mesh
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  USE_SOURCE_PERMISSIONS
)

install(FILES direct_servo.urdf leg.urdf simple_leg.urdf simple_servo_leg.urdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
