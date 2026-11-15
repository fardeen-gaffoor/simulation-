cmake_minimum_required(VERSION 3.16)
project(sandsim_plugin LANGUAGES CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(gz-sim8 REQUIRED)
find_package(gz-plugin2 REQUIRED)
find_package(gz-msgs10 REQUIRED)
find_package(gz-common5 REQUIRED)
find_package(sdformat14 REQUIRED)

include_directories(${gz-sim8_INCLUDE_DIRS}
                    ${gz-plugin2_INCLUDE_DIRS}
                    ${gz-msgs10_INCLUDE_DIRS}
                    ${gz-common5_INCLUDE_DIRS}
                    ${sdformat14_INCLUDE_DIRS})

add_library(sand_traction_plugin SHARED
  src/sand_traction_plugin.cpp
)

target_link_libraries(sand_traction_plugin
  gz-sim8::gz-sim8
  gz-plugin2::gz-plugin2
  gz-msgs10::gz-msgs10
  sdformat14::sdformat14
)

# Install the plugin library
install(TARGETS sand_traction_plugin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
