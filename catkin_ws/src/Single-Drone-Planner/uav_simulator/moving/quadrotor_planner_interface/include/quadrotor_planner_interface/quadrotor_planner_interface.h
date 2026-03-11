#ifndef __QUADROTOR_PLANNER_INTERFACE__
#define __QUADROTOR_PLANNER_INTERFACE__

#include "ros/ros.h"
#include <quadrotor_msgs/PositionCommand.h>
#include <nav_msgs/Odometry.h>
#include <Eigen/Core>

namespace quadrotor_planner_interface {
class QuadrotorPlannerInterface {
private:
  ros::Publisher ctrl_pub_;
  ros::Subscriber odom_sub_;
  ros::Timer plan_timer_, ctrl_timer_;
protected:
  int plan_hz_, ctrl_hz_;
  quadrotor_msgs::PositionCommand ctrl_msg_;
  virtual void plan_callback(const ros::TimerEvent& event);
  virtual void ctrl_callback(const ros::TimerEvent& event);
public:
  QuadrotorPlannerInterface(ros::NodeHandle& nh);
  ~QuadrotorPlannerInterface();
  // publish cmd
  void pubCmd(const Eigen::Vector3d& pos, const Eigen::Vector3d& vel, const Eigen::Vector3d& acc, double yaw = 0, double yaw_dot = 0);
  virtual bool getCmd(double t, Eigen::Vector3d& pos, Eigen::Vector3d& vel, Eigen::Vector3d& acc);
  virtual bool getCmd(double t, Eigen::Vector3d& pos, Eigen::Vector3d& vel, Eigen::Vector3d& acc, double& yaw, double& yaw_dot);
};
  
} // namespace quadrotor_planner_interface

#endif