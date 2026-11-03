cmake_minimum_required(VERSION 3.16)
project(rover_excavation_msgs)

# Module 6's status contract.  Deliberately separate from rover_navigation_msgs:
# the *call* into the excavation mechanism (the WorkCycle action) belongs to the
# caller (module 5), while everything the mechanism *reports* belongs here, so an
# operator display can read the berm status without depending on module 5.

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/MechanismStatus.msg"
  "msg/BermStatus.msg"
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
