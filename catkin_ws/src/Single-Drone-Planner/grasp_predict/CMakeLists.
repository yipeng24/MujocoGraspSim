cmake_minimum_required(VERSION 3.0.2)
project(yolo11_cup_seg)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  nav_msgs
  cv_bridge
  image_transport
  message_generation
)


generate_messages(DEPENDENCIES
  std_msgs
  sensor_msgs
  nav_msgs
)

catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    sensor_msgs
    nav_msgs
    cv_bridge
    image_transport
    message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS})

catkin_install_python(PROGRAMS
   scripts/cup_segentation_pc.py
   scripts/extract_images_from_bag.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )
