cmake_minimum_required(VERSION 3.16)
project(rover_dataset)

# rover_dataset is an *asset + tooling* package (ament_cmake), exactly like
# cat_arena and rover_description: the dataset definition, the labelling
# checklist and the generated dataset are data, and every executable is Python.
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# --------------------------------------------------------------------------- #
# python package: config, projection, labels, augment, scene, poseplan, capture,
# dataset, split, realphotos, finetune, preview, collector
# --------------------------------------------------------------------------- #
ament_python_install_package(${PROJECT_NAME})

# --------------------------------------------------------------------------- #
# command line tools -> lib/rover_dataset/<name>.py
# --------------------------------------------------------------------------- #
install(PROGRAMS
  scripts/collect_dataset.py
  scripts/collector_node.py
  scripts/import_real_photos.py
  scripts/split_dataset.py
  scripts/finetune.py
  scripts/verify_dataset.py
  scripts/preview_labels.py
  scripts/print_checklist.py
  DESTINATION lib/${PROJECT_NAME}
)

# --------------------------------------------------------------------------- #
# data: the dataset source of truth, the checklist, the launch files
# --------------------------------------------------------------------------- #
install(DIRECTORY
  config
  docs
  launch
  DESTINATION share/${PROJECT_NAME}
)

# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(rover_dataset_pytest test
    APPEND_ENV "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}"
    TIMEOUT 1200
  )
endif()

ament_package()
