cmake_minimum_required(VERSION 3.16)
project(rover_navigation)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# The wire contract this package speaks: module 5's own messages/actions, plus
# module 4's obstacle + localisation messages (the only obstacle knowledge the
# competition allows module 5 to have is live perception, so the dependency on
# rover_perception_msgs is structural, not incidental).
find_package(rover_navigation_msgs REQUIRED)
find_package(rover_perception_msgs REQUIRED)

ament_python_install_package(${PROJECT_NAME})

# Every node is a three-line shim in scripts/ calling into
# rover_navigation/nodes/*.py, so the installed executable is obvious and the
# node class stays importable (and testable) as a normal module.
install(PROGRAMS
  scripts/navigation.py
  scripts/velocity_gate.py
  scripts/mission.py
  scripts/navigation_info.py
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
  ament_add_pytest_test(rover_navigation_pytest test
    APPEND_ENV "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}"
    TIMEOUT 900
  )
endif()

ament_package()
