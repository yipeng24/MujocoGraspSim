cmake_minimum_required(VERSION 3.0.2)
project(drone_logger)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  message_generation
)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  DroneFullState.msg
)

## Generate services in the 'srv' folder
add_service_files(
  FILES
  SetPayload.srv
)

## Generate added messages and services
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs geometry_msgs message_runtime
)

## Mark python scripts as executable
catkin_install_python(PROGRAMS
  scripts/payload_cli.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
