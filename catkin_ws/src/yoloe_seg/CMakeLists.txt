cmake_minimum_required(VERSION 3.0.2)
project(yoloe_seg)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  nav_msgs
  std_msgs
  quadrotor_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/yoloe_seg_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY cfg launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
