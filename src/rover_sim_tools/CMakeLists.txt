cmake_minimum_required(VERSION 3.16)
project(rover_sim_tools)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Python package: motor_battery_model node + shared motor/battery models
ament_python_install_package(${PROJECT_NAME})

# Command line tools -> lib/rover_sim_tools/<name>.py
install(PROGRAMS
  scripts/motor_battery_model.py
  DESTINATION lib/${PROJECT_NAME}
)

# Tests
if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(rover_sim_tools_pytest test
    APPEND_ENV "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}"
    TIMEOUT 900
  )
endif()

ament_package()
