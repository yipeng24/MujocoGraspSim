cmake_minimum_required(VERSION 2.8.3)
project(fake_rc)

set(CMAKE_BUILD_TYPE "Release")
ADD_COMPILE_OPTIONS(-std=c++11 )
ADD_COMPILE_OPTIONS(-std=c++14 )
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  mavros
  std_msgs
  sensor_msgs
  mav_msgs
)
find_package(Eigen3 REQUIRED)

catkin_package(
 CATKIN_DEPENDS roscpp std_msgs mavros sensor_msgs mav_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

add_executable (fake_rc_node src/fake_rc.cpp)
target_link_libraries(fake_rc_node
  ${catkin_LIBRARIES})
