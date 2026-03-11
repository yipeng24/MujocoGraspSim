cmake_minimum_required(VERSION 3.0.2)
project(dynamic_planner)

add_compile_options(-std=c++14)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

find_package(catkin REQUIRED COMPONENTS
  nav_msgs
  pcl_ros
  sensor_msgs
  cv_bridge
  quadrotor_planner_interface
)
find_package(PCL REQUIRED)
message(${catkin_INCLUDE_DIRS})

catkin_package(
 INCLUDE_DIRS 
 include
#  LIBRARIES map_free_planner
 CATKIN_DEPENDS
  cv_bridge
  pcl_ros
  sensor_msgs
  quadrotor_planner_interface
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/dynamic_planner.cc
)
target_link_libraries( ${PROJECT_NAME} ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(${PROJECT_NAME}_test_node
  src/test_node.cpp
)

target_link_libraries(${PROJECT_NAME}_test_node
  ${PROJECT_NAME}
  ${PCL_LIBRARIES}
)
