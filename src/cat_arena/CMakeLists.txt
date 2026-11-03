cmake_minimum_required(VERSION 3.16)
project(cat_arena)

# cat_arena is a Python + data/asset package: the generator, the nodes and the
# verification scripts are Python, and config/, launch/ and assets/ are pure
# data.  There is deliberately no C++ here - the later modules (rover_excavation)
# are the ones that need ament_cmake C++ targets.
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# --------------------------------------------------------------------------- #
# python package (cat_arena/config.py, terrain.py, generate.py, ...)
# --------------------------------------------------------------------------- #
ament_python_install_package(${PROJECT_NAME})

# --------------------------------------------------------------------------- #
# command line tools -> lib/cat_arena/<name>.py  (ros2 run cat_arena <name>.py)
# --------------------------------------------------------------------------- #
install(PROGRAMS
  scripts/env_check.py
  scripts/generate_arena.py
  scripts/publish_start_pose.py
  scripts/render_overview.py
  DESTINATION lib/${PROJECT_NAME}
)

# --------------------------------------------------------------------------- #
# data: the single-source-of-truth config, launch files, marker/spawn assets
# --------------------------------------------------------------------------- #
install(DIRECTORY
  config
  launch
  assets
  DESTINATION share/${PROJECT_NAME}
)

# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  # PYTHONPATH gets the source tree so the tests can `import cat_arena` even
  # before/without sourcing the install space.
  ament_add_pytest_test(cat_arena_pytest test
    APPEND_ENV "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}"
    TIMEOUT 900
  )
endif()

ament_package()
