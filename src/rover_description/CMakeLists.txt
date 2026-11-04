cmake_minimum_required(VERSION 3.16)
project(rover_description)

# rover_description is an *asset* package (ament_cmake): the xacro model, the
# generated URDF/SDF, the calibration file and the mass/envelope artifacts are
# data, and the one small runtime node (stub_topics) is Python - the whole stack
# is Python.  There is deliberately no C++ target here.
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# --------------------------------------------------------------------------- #
# python package: config, geometry, calibration, mass, bridges, generate, stub
# --------------------------------------------------------------------------- #
ament_python_install_package(${PROJECT_NAME})

# --------------------------------------------------------------------------- #
# command line tools -> lib/rover_description/<name>.py
#   ros2 run rover_description generate_robot
#   ros2 run rover_description check_envelope
#   ros2 run rover_description mass_budget
#   ros2 run rover_description stub_topics.py
# --------------------------------------------------------------------------- #
install(PROGRAMS
  scripts/generate_robot.py
  scripts/check_envelope.py
  scripts/mass_budget.py
  scripts/simple_drive.py
  scripts/stub_topics.py
  DESTINATION lib/${PROJECT_NAME}
)

# --------------------------------------------------------------------------- #
# data: the source of truth (rover.yaml, bridges.yaml), the model, launch + rviz
# --------------------------------------------------------------------------- #
install(DIRECTORY
  config
  urdf
  launch
  rviz
  DESTINATION share/${PROJECT_NAME}
)

# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  # PYTHONPATH gets the source tree so the tests can `import rover_description`
  # even before/without sourcing the install space.
  ament_add_pytest_test(rover_description_pytest test
    APPEND_ENV "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}"
    TIMEOUT 900
  )
endif()

ament_package()
