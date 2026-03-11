cmake_minimum_required(VERSION 3.0.2)
project(mavros_sim)

find_package(catkin REQUIRED COMPONENTS
  rospy
  mavros_msgs
  sensor_msgs
  nav_msgs
)

catkin_package()

catkin_install_python(PROGRAMS scripts/mavros_sim.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
