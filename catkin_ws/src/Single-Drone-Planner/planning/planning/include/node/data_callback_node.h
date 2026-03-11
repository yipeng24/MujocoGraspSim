#ifndef DATA_CALLBACK
#define DATA_CALLBACK

#include "util_gym/util_gym.hpp"
#include "util_gym/data_manager.hpp"
#include "rotation_util/rotation_util.hpp"
#include "visualization_interface/vis_interface.h"
#include <thread>

#ifdef ROS
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <quadrotor_msgs/OccMap3d.h>
#include <sensor_msgs/Joy.h>
#include <sensor_msgs/PointCloud2.h>
#include <quadrotor_msgs/UAMFullState.h>
#endif

class DataCallBacks{
 public:
  // outer ptr
  std::shared_ptr<ShareDataManager> dataManagerPtr_;
  std::shared_ptr<parameter_server::ParaeterSerer> paraPtr_;
  std::shared_ptr<vis_interface::VisInterface> vis_ptr_;

 private:

  /*********** Ros Variables ***************/
  #ifdef ROS
  ros::Subscriber odom_sub_, plan_trigger_sub_, target_sub_, land_triger_sub_, gridmap_sub_, goal_sub_, joy_sub_, goal_full_state_sub_;
  #endif
  /****************************************/

  /*********** SSDBUS Variables ***************/

  /****************************************/

 public:
  DataCallBacks(std::shared_ptr<ShareDataManager> dataManagerPtr, std::shared_ptr<parameter_server::ParaeterSerer> paraPtr);

  inline void set_vis_ptr(std::shared_ptr<vis_interface::VisInterface> vis_ptr){
    vis_ptr_ = vis_ptr;
  }

  /*********** ROS Functions ***************/
  #ifdef ROS
  void init_ros(ros::NodeHandle& nh);
  void odom_callback(const nav_msgs::Odometry::ConstPtr& msgPtr);
  void triger_plan_callback(const geometry_msgs::PoseStampedConstPtr& msgPtr);
  void land_triger_callback(const geometry_msgs::PoseStampedConstPtr& msgPtr);
  void target_callback(const nav_msgs::Odometry::ConstPtr& msgPtr);
  void goal_callback(const geometry_msgs::PoseStampedConstPtr& msgPtr);
  void goal_full_state_callback(const quadrotor_msgs::UAMFullStateConstPtr& msgPtr);
  // void gridmap_callback(const quadrotor_msgs::OccMap3dConstPtr& msgPtr);
  void gridmap_callback(const sensor_msgs::PointCloud2ConstPtr& msgPtr);
  void joy_callback(const sensor_msgs::Joy::ConstPtr& msgPtr);

  #endif
  /****************************************/

  /*********** SSDBUS Functions ***************/
  #ifdef SS_DBUS

  #endif
  /****************************************/

  /*********** Share Functions ***************/

  /****************************************/

};

#endif