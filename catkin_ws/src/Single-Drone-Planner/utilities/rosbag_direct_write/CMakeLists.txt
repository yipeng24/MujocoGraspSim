cmake_minimum_required(VERSION 2.8.3)
project(rosbag_direct_write)

find_package(catkin REQUIRED COMPONENTS cpp_common rosbag_storage roscpp roscpp_serialization)
find_package(Eigen3 REQUIRED)


catkin_package(
	INCLUDE_DIRS include ${EIGEN3_INCLUDE_DIR}
	CATKIN_DEPENDS cpp_common rosbag_storage roscpp roscpp_serialization
	LIBRARIES rosbag_direct_write
  DEPENDS EIGEN3
)

include_directories(include ${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

add_library(${PROJECT_NAME}
  src/direct_bag.cpp)


add_executable(test_bag src/test_direct_bag.cpp)
target_link_libraries(test_bag
  rosbag_direct_write 
  ${catkin_LIBRARIES}
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if(CATKIN_ENABLE_TESTING)
  find_package(sensor_msgs REQUIRED)
  list(APPEND catkin_LIBRARIES ${sensor_msgs_LIBRARIES})
  include_directories(include ${catkin_INCLUDE_DIRS} ${sensor_msgs_INCLUDE_DIRS})
  # Make sure include_directories and catkin_LIBRARIES are setup before adding
  add_subdirectory(test)
endif()

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)
