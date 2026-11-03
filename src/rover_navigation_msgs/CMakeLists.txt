cmake_minimum_required(VERSION 3.16)
project(rover_navigation_msgs)

# Module 5's wire contract.  Everything else in rover_navigation is Python; the
# interfaces live in their own package so that module 6 (rover_excavation) and
# any operator display can depend on the goal/status contract without pulling in
# the navigation implementation.

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/NavigationStatus.msg"
  "msg/GateStatus.msg"
  "msg/MissionStatus.msg"
  "action/NavigateTo.action"
  "action/WorkCycle.action"
  DEPENDENCIES std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
