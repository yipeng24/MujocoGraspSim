cmake_minimum_required(VERSION 3.0.2)
project(task_manager)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  quadrotor_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/occlusion_analyzer.py
  scripts/task_fsm.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY cfg launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
