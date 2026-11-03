cmake_minimum_required(VERSION 3.16)
project(rover_perception)

# rover_perception is Python: the algorithms live in rover_perception/*.py, the
# rclpy nodes in scripts/, and config/, launch/ and docs/ are pure data.  The
# only C++ in module 4 is the message package (rover_perception_msgs).

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rover_perception_msgs REQUIRED)

ament_python_install_package(${PROJECT_NAME})

# --------------------------------------------------------------------------- #
# rclpy nodes and command line tools -> lib/rover_perception/<name>.py
# --------------------------------------------------------------------------- #
install(PROGRAMS
  scripts/camera_manager.py
  scripts/tag_detector.py
  scripts/tag_localizer.py
  scripts/obstacle_detector.py
  scripts/health_monitor.py
  scripts/generate_tags.py
  scripts/preview_tags.py
  scripts/perception_info.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config
  launch
  docs
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(rover_perception_pytest test
    APPEND_ENV "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}"
    TIMEOUT 900
  )
endif()

ament_package()
