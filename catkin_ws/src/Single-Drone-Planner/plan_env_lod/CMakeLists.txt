cmake_minimum_required(VERSION 2.8.3)
project(plan_env_lod)

set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -Wall")


if(NOT OpenCV_DIR)
  foreach(_hint
    /usr/lib/x86_64-linux-gnu/cmake/opencv4
    /usr/share/opencv4
    /opt/conda/envs/mujoco310/lib/cmake/opencv4
  )
    if(EXISTS "${_hint}/OpenCVConfig.cmake")
      set(OpenCV_DIR "${_hint}" CACHE PATH "OpenCV cmake dir")
      break()
    endif()
  endforeach()
endif()
find_package(OpenCV REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  visualization_msgs
  cv_bridge
  message_filters
  pcl_conversions
  pcl_ros
  quadrotor_msgs
)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.7 REQUIRED)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES plan_env_lod
 CATKIN_DEPENDS roscpp std_msgs quadrotor_msgs 
#  DEPENDS system_lib
)

include_directories( 
    SYSTEM 
    include 
    ${catkin_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS} 
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})

add_library( plan_env_lod 
    src/sdf_map.cpp 
    src/obj_predictor.cpp 
    src/edt_environment.cpp 
    src/raycast.cpp
    src/map_ros.cpp
    )
target_link_libraries( plan_env_lod
    ${catkin_LIBRARIES} 
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
    )  

# add_executable(obj_generator
#     src/obj_generator.cpp 
#     )
# target_link_libraries(obj_generator 
#     ${catkin_LIBRARIES})

############## MAIN node #############
# w/o ros
# add_executable(gen_occmap_node
#     src/gen_occmap_node.cpp
#     )
# target_link_libraries(gen_occmap_node
#     plan_env_lod
#     ${catkin_LIBRARIES} 
#     ${PCL_LIBRARIES}
#     )
# add_dependencies(gen_occmap_node ${catkin_EXPORTED_TARGETS})

# w/ ros
add_executable(gen_occmap_node
    src/lod_map_node.cpp
    )
target_link_libraries(gen_occmap_node
    plan_env_lod
    ${catkin_LIBRARIES} 
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
    )
add_dependencies(gen_occmap_node ${catkin_EXPORTED_TARGETS})
