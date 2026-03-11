#include "quadrotor_planner_interface/quadrotor_planner_interface.h"

namespace quadrotor_planner_interface {
  QuadrotorPlannerInterface::QuadrotorPlannerInterface(ros::NodeHandle& nh) {
    // parameters
    nh.param("planning/plan_hz", plan_hz_, 1);
    nh.param("planning/ctrl_hz", ctrl_hz_, 100);
    // ros publishers
    ctrl_pub_ = nh.advertise<quadrotor_msgs::PositionCommand>("position_cmd", 1);
    plan_timer_ = nh.createTimer(ros::Duration(1.0/plan_hz_), &QuadrotorPlannerInterface::plan_callback, this);
    ctrl_timer_ = nh.createTimer(ros::Duration(1.0/ctrl_hz_), &QuadrotorPlannerInterface::ctrl_callback, this);
  }

  QuadrotorPlannerInterface::~QuadrotorPlannerInterface() {
  }
  void QuadrotorPlannerInterface::pubCmd(
   const Eigen::Vector3d& pos,
   const Eigen::Vector3d& vel,
   const Eigen::Vector3d& acc,
   const double yaw/*  = 0 */,
   const double yaw_dot/* =0 */) {
    geometry_msgs::Point tmpPoint;
    geometry_msgs::Vector3 tmpVector;
    tmpPoint.x = pos.x();
    tmpPoint.y = pos.y();
    tmpPoint.z = pos.z();
    ctrl_msg_.position = tmpPoint;
    tmpVector.x = vel.x();
    tmpVector.y = vel.y();
    tmpVector.z = vel.z();
    ctrl_msg_.velocity = tmpVector;
    tmpVector.x = acc.x();
    tmpVector.y = acc.y();
    tmpVector.z = acc.z();
    ctrl_msg_.acceleration = tmpVector;
    ctrl_msg_.yaw = yaw;
    ctrl_msg_.yaw_dot = yaw_dot;
    ctrl_msg_.header.stamp = ros::Time::now();
    ctrl_msg_.trajectory_id ++;
    ctrl_pub_.publish(ctrl_msg_);
  }
  bool QuadrotorPlannerInterface::getCmd(double t, Eigen::Vector3d& pos, Eigen::Vector3d& vel, Eigen::Vector3d& acc) {
    return false;
  }
  bool QuadrotorPlannerInterface::getCmd(double t, Eigen::Vector3d& pos, Eigen::Vector3d& vel, Eigen::Vector3d& acc, double& yaw, double& yaw_dot) {
    return false;
  }
  void QuadrotorPlannerInterface::plan_callback(const ros::TimerEvent& event) {

  };
  void QuadrotorPlannerInterface::ctrl_callback(const ros::TimerEvent& event) {

  };
} // namespace quadrotor_planner_interface
