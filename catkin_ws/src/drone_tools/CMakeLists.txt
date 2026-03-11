cmake_minimum_required(VERSION 3.0.2)
project(drone_tools)

find_package(catkin REQUIRED COMPONENTS
  rospy
  quadrotor_msgs
  nav_msgs
  sensor_msgs
  geometry_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/mujoco_goal_picker.py
  scripts/joy_to_position_cmd.py
  scripts/key_to_joy_UAM.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
