cmake_minimum_required(VERSION 2.8.3)
project(pcd_to_pcl)

set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_CXX_FLAGS "-std=c++11")
ADD_COMPILE_OPTIONS(-std=c++11 )
ADD_COMPILE_OPTIONS(-std=c++14 )
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  geometry_msgs
  tf
)
find_package(Eigen3 REQUIRED)

catkin_package()

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

catkin_install_python(PROGRAMS scripts/pub_pcl_from_pcd.py
  scripts/save_pcd_pose_to_yaml.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
