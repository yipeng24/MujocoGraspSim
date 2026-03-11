cmake_minimum_required(VERSION 2.8.3)
project(ch_rc_sdf)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -Wall")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  quadrotor_msgs
  nav_msgs
  sensor_msgs
  pcl_conversions
  pcl_ros
)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.7 REQUIRED)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES ch_rc_sdf
 CATKIN_DEPENDS roscpp std_msgs quadrotor_msgs nav_msgs sensor_msgs
#  DEPENDS system_lib
)

include_directories( 
    SYSTEM 
    include 
    ${catkin_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS} 
    ${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})


add_library( ch_rc_sdf 
    src/ch_rc_sdf.cpp 
    src/vis_ch_rc_sdf.cpp 
    )
target_link_libraries( ch_rc_sdf
    ${catkin_LIBRARIES} 
    ${PCL_LIBRARIES}
    )  

############## MAIN node #############
add_executable(test_node src/test_node.cpp)
target_link_libraries(test_node
  ch_rc_sdf
  ${catkin_LIBRARIES}
)
add_dependencies(test_node ${catkin_EXPORTED_TARGETS})