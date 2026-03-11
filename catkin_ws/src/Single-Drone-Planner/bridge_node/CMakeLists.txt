cmake_minimum_required(VERSION 2.8.3)
project(bridge_node)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -Wall")

find_package(OpenCV REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  quadrotor_msgs
  cv_bridge
  nav_msgs
  grasp_chassis_comm
  message_filters
)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.7 REQUIRED)

catkin_package(
 INCLUDE_DIRS 
 LIBRARIES 
 CATKIN_DEPENDS roscpp std_msgs quadrotor_msgs nav_msgs grasp_chassis_comm message_filters
#  DEPENDS system_lib
)

include_directories( 
    SYSTEM 
    include 
    ${catkin_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS} 
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})

############## MAIN node #############
add_executable(bridge_node src/bridge_node.cpp)
target_link_libraries(bridge_node
    ${PCL_LIBRARIES}
    ${catkin_LIBRARIES}
)
add_dependencies(bridge_node ${catkin_EXPORTED_TARGETS})