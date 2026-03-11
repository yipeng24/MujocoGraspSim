cmake_minimum_required(VERSION 3.0.2)
project(quadrotor_planner_interface)

add_compile_options(-std=c++14)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  quadrotor_msgs
  message_filters
)

find_package(Eigen3)

catkin_package(
 INCLUDE_DIRS include ${EIGEN3_INCLUDE_DIR}
 LIBRARIES ${PROJECT_NAME}_lib
 CATKIN_DEPENDS quadrotor_msgs roscpp
)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}_lib
  src/${PROJECT_NAME}.cc
)

target_link_libraries(${PROJECT_NAME}_lib
  ${catkin_LIBRARIES}
)

add_executable(${PROJECT_NAME}_test_node
 src/test_node.cpp
)

target_link_libraries(${PROJECT_NAME}_test_node
 ${catkin_LIBRARIES} ${PROJECT_NAME}_lib
)
