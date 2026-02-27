cmake_minimum_required(VERSION 3.0.2)
project(beetle_omni)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++17)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(Eigen3 REQUIRED)

## System dependencies are found with CMake's conventions
find_package(catkin REQUIRED COMPONENTS
        aerial_robot_control
        aerial_robot_estimation
        aerial_robot_model
        aerial_robot_msgs
        diagnostic_msgs
        message_generation
        pluginlib
        roscpp
        std_srvs
)

# Use $ENV{HOME} rather than '~' to get a portable absolute path.
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(ONNXRUNTIME_DIR "$ENV{HOME}/ros/tau_ws/libraries/onnxruntime-linux-aarch64-1.22.0")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" OR CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "amd64")
  set(ONNXRUNTIME_DIR "$ENV{HOME}/libraries/onnxruntime")
else()
  message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}. Please set ONNXRUNTIME_DIR manually in CMakeLists.txt")
endif()

###################################
## catkin specific configuration ##
###################################

add_service_files(FILES ReloadPolicy.srv)
generate_messages(DEPENDENCIES std_msgs)

catkin_package(
  CATKIN_DEPENDS aerial_robot_control aerial_robot_estimation aerial_robot_model aerial_robot_msgs
                 message_runtime pluginlib roscpp spinal std_srvs
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${ONNXRUNTIME_DIR}/include
  )

link_directories(${ONNXRUNTIME_DIR}/lib)
add_executable(beetle_deploy src/beetle_deploy.cpp)
target_include_directories(beetle_deploy PRIVATE ${catkin_INCLUDE_DIRS})
target_link_libraries(beetle_deploy 
  ${catkin_LIBRARIES}
  ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
  )

add_executable(gimbal_response src/gimbal_response.cpp)
target_include_directories(gimbal_response PRIVATE ${catkin_INCLUDE_DIRS})
target_link_libraries(gimbal_response ${catkin_LIBRARIES})

add_library(beetle_pose_rlagent src/beetle_pose_rlagent.cpp)
add_dependencies(beetle_pose_rlagent
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(beetle_pose_rlagent
  ${catkin_LIBRARIES}
  ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
  )

install(DIRECTORY config plugins launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  USE_SOURCE_PERMISSIONS
  )

# if(CATKIN_ENABLE_TESTING)
#   find_package(rostest REQUIRED)
#   add_subdirectory(test)
# endif()
