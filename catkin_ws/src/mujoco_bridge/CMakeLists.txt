
cmake_minimum_required(VERSION 3.0.2)
project(mujoco_bridge)

add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  nav_msgs
  sensor_msgs
)

catkin_package()

# -----------------------------
# MuJoCo path (configurable)
# -----------------------------
# Priority:
# 1) -DMUJOCO_DIR=... (cmake arg)
# 2) Environment MUJOCO_DIR
# 3) Fallback to your conda mujoco package path (edit if needed)
set(MUJOCO_DIR "" CACHE PATH "MuJoCo root directory (contains include/ and libmujoco.so*)")

if(NOT MUJOCO_DIR)
  if(DEFINED ENV{MUJOCO_DIR})
    set(MUJOCO_DIR $ENV{MUJOCO_DIR})
  else()
    set(MUJOCO_DIR "/opt/conda/envs/mujoco310/lib/python3.10/site-packages/mujoco")
  endif()
endif()

# Header path
set(MUJOCO_INCLUDE_DIR "${MUJOCO_DIR}/include")

# Library: try common layouts:
#   - ${MUJOCO_DIR}/lib/libmujoco.so*
#   - ${MUJOCO_DIR}/libmujoco.so*
find_library(MUJOCO_LIBRARY
  NAMES mujoco libmujoco libmujoco.so libmujoco.so.3 libmujoco.so.3.5.0
  PATHS "${MUJOCO_DIR}/lib" "${MUJOCO_DIR}"
  NO_DEFAULT_PATH
)

if(NOT EXISTS "${MUJOCO_INCLUDE_DIR}/mujoco/mujoco.h")
  message(FATAL_ERROR "MuJoCo header not found: ${MUJOCO_INCLUDE_DIR}/mujoco/mujoco.h (MUJOCO_DIR=${MUJOCO_DIR})")
endif()

if(NOT MUJOCO_LIBRARY)
  message(FATAL_ERROR "MuJoCo library not found under ${MUJOCO_DIR} (looked in ${MUJOCO_DIR} and ${MUJOCO_DIR}/lib).")
endif()

message(STATUS "Using MUJOCO_DIR=${MUJOCO_DIR}")
message(STATUS "MuJoCo include: ${MUJOCO_INCLUDE_DIR}")
message(STATUS "MuJoCo library: ${MUJOCO_LIBRARY}")

include_directories(
  ${MUJOCO_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
)

#add_executable(test_node src/test_node.cpp)

#target_link_libraries(test_node
#  ${catkin_LIBRARIES}
#  ${MUJOCO_LIBRARY}
#)
