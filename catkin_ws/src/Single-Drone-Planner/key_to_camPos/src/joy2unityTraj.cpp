/*
  MIT License

  Copyright (c) 2021 Hongkai Ye (kyle_yeh@163.com, hkye@zju.edu.cn)

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
*/
#include <Eigen/Eigen>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Joy.h>
#include "tf/tf.h"
#include <trajectory_msgs/MultiDOFJointTrajectory.h>
#include <trajectory_msgs/MultiDOFJointTrajectoryPoint.h>

#define CH_FORWARD 0
#define CH_LEFT 1
#define CH_UP 2
#define CH_PITCH 3
#define CH_YAW 4

ros::Publisher unityTraj_pub_, desired_odom_pub_;
ros::Subscriber joy_sub_;
ros::Timer unityTraj_pub_timer_;
int unityTraj_pub_rate_ = 50.0;
double unityTraj_pub_cyc_;
nav_msgs::Odometry unity_traj_msg_;
sensor_msgs::Joy rcv_joy_msg_;
bool have_joy_ = false;
bool pub_cam_ = true;
Eigen::Vector3d pos_;
Eigen::Quaterniond q_;
double pitch_=0, yaw_=0;

void angLimit(double& ang)
{
  ang = ang > M_PI ? ang - 2*M_PI : ang;
  ang = ang <-M_PI ? ang + 2*M_PI : ang;
}

trajectory_msgs::MultiDOFJointTrajectory create_unity_trajectory(const Eigen::Vector3d& pos, 
                                                                const Eigen::Vector3d& vel, 
                                                                const Eigen::Vector3d& acc,
                                                                const Eigen::Quaterniond& q) {
    trajectory_msgs::MultiDOFJointTrajectory traj_msg;
    trajectory_msgs::MultiDOFJointTrajectoryPoint traj_point;

    // Set transform (position + orientation)
    geometry_msgs::Transform transform;
    transform.translation.x = pos.x();
    transform.translation.y = pos.y();
    transform.translation.z = pos.z();

    // Convert yaw to quaternion using tf
    geometry_msgs::Quaternion quat;
    quat.x = q.x();
    quat.y = q.y();
    quat.z = q.z();
    quat.w = q.w();
    transform.rotation = quat;
    traj_point.transforms.push_back(transform);


    geometry_msgs::Twist velocity;
    velocity.linear.x = vel.x();
    velocity.linear.y = vel.y();
    velocity.linear.z = vel.z();
    traj_point.velocities.push_back(velocity);


    // Zero acceleration
    geometry_msgs::Twist acceleration;
    acceleration.linear.x = acc.x();
    acceleration.linear.y = acc.y();
    acceleration.linear.z = acc.z();
    traj_point.accelerations.push_back(acceleration);


    // Add point to trajectory message
    traj_msg.points.push_back(traj_point);
    traj_msg.header.stamp = ros::Time::now();


    return traj_msg;
}


void rcvJoyCallbck(const sensor_msgs::Joy &joy_msg)
{
  rcv_joy_msg_ = joy_msg;
  have_joy_ = true;
  return;
}

void unityTrajPubCallbck(const ros::TimerEvent& event){

  if(!have_joy_) return;

  unity_traj_msg_.header.stamp = ros::Time::now();

  Eigen::Vector3d T_b,T_w; //单位时间平移
  T_b << rcv_joy_msg_.axes.at(CH_FORWARD), rcv_joy_msg_.axes.at(CH_LEFT), rcv_joy_msg_.axes.at(CH_UP);
  T_b*=unityTraj_pub_cyc_;

  Eigen::Matrix2d R_b2w_2d;
  R_b2w_2d << cos(yaw_),-sin(yaw_),
              sin(yaw_), cos(yaw_);

  T_w.head(2) = R_b2w_2d*T_b.head(2);
  T_w.z() = T_b.z();

  pos_.x() += T_w.x();
  pos_.y() += T_w.y();
  pos_.z() += T_w.z();

  pitch_ += unityTraj_pub_cyc_*rcv_joy_msg_.axes.at(CH_PITCH);
  yaw_ += unityTraj_pub_cyc_*rcv_joy_msg_.axes.at(CH_YAW);
  angLimit(pitch_);
  angLimit(yaw_);
  q_ = Eigen::AngleAxisd(yaw_, Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(pitch_, Eigen::Vector3d::UnitY()) ;

  auto traj_msg = create_unity_trajectory(pos_, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), q_);
  unityTraj_pub_.publish(traj_msg);

  //! pub desied odom
  nav_msgs::Odometry desired_odom_msg;
  desired_odom_msg.header.stamp = ros::Time::now();
  desired_odom_msg.header.frame_id = "world";

  desired_odom_msg.pose.pose.position.x = pos_.x();
  desired_odom_msg.pose.pose.position.y = pos_.y();
  desired_odom_msg.pose.pose.position.z = pos_.z();

  desired_odom_msg.pose.pose.orientation.x = q_.x();
  desired_odom_msg.pose.pose.orientation.y = q_.y();
  desired_odom_msg.pose.pose.orientation.z = q_.z();
  desired_odom_msg.pose.pose.orientation.w = q_.w();

  desired_odom_pub_.publish(desired_odom_msg);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "joy2unityTraj");
  ros::NodeHandle nh("~");

  nh.getParam("pos_init_x", pos_.x());
  nh.getParam("pos_init_y", pos_.y());
  nh.getParam("pos_init_z", pos_.z());
  nh.getParam("yaw_init", yaw_);

  std::cout << "pos_init: " << pos_.transpose() << std::endl;
  std::cout << "yaw_init: " << yaw_ << std::endl;

  joy_sub_ = nh.subscribe("/joy", 1, rcvJoyCallbck);
  unityTraj_pub_ = nh.advertise<trajectory_msgs::MultiDOFJointTrajectory>("unity_traj_cmd", 1);
  desired_odom_pub_ = nh.advertise<nav_msgs::Odometry>("desired_odom", 1);

  unityTraj_pub_cyc_ = 1./(double)unityTraj_pub_rate_;
  unityTraj_pub_timer_ = nh.createTimer(ros::Duration(unityTraj_pub_cyc_),&unityTrajPubCallbck);

  ros::spin();
}
