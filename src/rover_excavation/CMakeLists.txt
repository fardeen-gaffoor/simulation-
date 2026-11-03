cmake_minimum_required(VERSION 3.16)
project(rover_excavation)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# The wire contract module 6 speaks: module 5's WorkCycle action (module 5 is the
# caller, so the contract had to live there), module 6's own status messages, and
# the ordinary message packages the node publishes and subscribes to.
find_package(rover_navigation_msgs REQUIRED)
find_package(rover_excavation_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

ament_python_install_package(${PROJECT_NAME})

# Every node is a three-line shim in scripts/ calling into
# rover_excavation/nodes/*.py, so the installed executable is obvious and the
# node class stays importable (and testable) as a normal module.
install(PROGRAMS
  scripts/excavation.py
  scripts/excavation_info.py
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
  ament_add_pytest_test(rover_excavation_pytest test
    APPEND_ENV "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}"
    TIMEOUT 900
  )
endif()

ament_package()
