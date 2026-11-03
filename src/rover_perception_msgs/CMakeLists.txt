cmake_minimum_required(VERSION 3.16)
project(rover_perception_msgs)

# Module 4's wire contract.  Everything else in rover_perception is Python; the
# messages live in their own package so that module 5 (rover_navigation) and,
# later, rover_camera_driver can depend on the interfaces without pulling in the
# perception implementation.

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/CameraPhase.msg"
  "msg/CameraStatus.msg"
  "msg/CameraStatusArray.msg"
  "msg/Obstacle.msg"
  "msg/ObstacleArray.msg"
  "msg/TagDetection.msg"
  "msg/TagDetectionArray.msg"
  "msg/LocalizationState.msg"
  "msg/PerceptionHealth.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
