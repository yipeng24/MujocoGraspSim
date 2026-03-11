cmake_minimum_required(VERSION 2.8.3)
project(key_to_camPos)
# set(CMAKE_CXX_STANDARD 14)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -Wall")
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR})

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    nav_msgs
    sensor_msgs
    tf
    geometry_msgs
    quadrotor_msgs
    )

find_package(Eigen3 REQUIRED)

catkin_package(
    # DEPENDS Eigen
    CATKIN_DEPENDS roscpp nav_msgs sensor_msgs tf geometry_msgs quadrotor_msgs
)

include_directories("/usr/include/eigen3")

include_directories(
  SYSTEM 
  ${catkin_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
)

add_executable(
  joy2camPos
  # src/joy2unityTraj.cpp
  src/joy2UAM.cpp
)

target_link_libraries( 
  joy2camPos
  ${catkin_LIBRARIES}
)

# catkin_install_python(PROGRAMS scripts/key_to_joy.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

# catkin_install_python(PROGRAMS scripts/key_to_joy_free.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

catkin_install_python(PROGRAMS scripts/key_to_joy_UAM.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

