cmake_minimum_required(VERSION 3.0.2)
project(mujoco_perception)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  nav_msgs
  std_msgs
  quadrotor_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/mujoco_gt_seg_pub.py
  scripts/mujoco_rgb_depth_pub.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY cfg launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
