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

#define CH_FORWARD 0
#define CH_LEFT 1
#define CH_UP 2
#define CH_PITCH 3
#define CH_YAW 4

ros::Publisher camPos_pub_;
ros::Subscriber joy_sub_;
ros::Timer camPos_pub_timer_;
int camPos_pub_rate_ = 50.0;
double camPos_pub_cyc_;
nav_msgs::Odometry cam_pos_msg_;
sensor_msgs::Joy rcv_joy_msg_;
bool have_joy_ = false;
bool pub_cam_ = true;

double pitch_=0, yaw_=0;

void angLimit(double& ang)
{
  ang = ang > M_PI ? ang - 2*M_PI : ang;
  ang = ang <-M_PI ? ang + 2*M_PI : ang;
}

void rcvJoyCallbck(const sensor_msgs::Joy &joy_msg)
{
  rcv_joy_msg_ = joy_msg;
  have_joy_ = true;
  return;
}

void camPosPubCallbck(const ros::TimerEvent& event){

  if(!have_joy_) return;

  cam_pos_msg_.header.stamp = ros::Time::now();

  Eigen::Vector3d T_b,T_w; //单位时间平移
  T_b << rcv_joy_msg_.axes.at(CH_FORWARD), rcv_joy_msg_.axes.at(CH_LEFT), rcv_joy_msg_.axes.at(CH_UP);
  T_b*=camPos_pub_cyc_;

  Eigen::Matrix2d R_b2w_2d;
  R_b2w_2d << cos(yaw_),-sin(yaw_),
              sin(yaw_), cos(yaw_);

  T_w.head(2) = R_b2w_2d*T_b.head(2);
  T_w.z() = T_b.z();

  cam_pos_msg_.pose.pose.position.x += T_w.x();
  cam_pos_msg_.pose.pose.position.y += T_w.y();
  cam_pos_msg_.pose.pose.position.z += T_w.z();

  pitch_ += camPos_pub_cyc_*rcv_joy_msg_.axes.at(CH_PITCH);
  yaw_ += camPos_pub_cyc_*rcv_joy_msg_.axes.at(CH_YAW);
  angLimit(pitch_);
  angLimit(yaw_);
  Eigen::Quaterniond q_b2w,q_c2b,q_c2w;
  q_b2w = Eigen::AngleAxisd(yaw_, Eigen::Vector3d::UnitZ()) *
          Eigen::AngleAxisd(pitch_, Eigen::Vector3d::UnitY()) ;

  q_c2b.w() =-0.5;
  q_c2b.x() = 0.5;
  q_c2b.y() =-0.5;
  q_c2b.z() = 0.5;

  q_c2w = pub_cam_ ? q_b2w*q_c2b : q_b2w;

  cam_pos_msg_.pose.pose.orientation.w = q_c2w.w();
  cam_pos_msg_.pose.pose.orientation.x = q_c2w.x();
  cam_pos_msg_.pose.pose.orientation.y = q_c2w.y();
  cam_pos_msg_.pose.pose.orientation.z = q_c2w.z();

  camPos_pub_.publish(cam_pos_msg_);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "joy2camPos");
  ros::NodeHandle nh("~");

  nh.getParam("pub_cam", pub_cam_);
  

  cam_pos_msg_.header.frame_id = "world";
  cam_pos_msg_.pose.pose.position.x = 0.0;
  cam_pos_msg_.pose.pose.position.y = 0.0;
  cam_pos_msg_.pose.pose.position.z = 1.0;
  cam_pos_msg_.pose.pose.orientation.w = 1.0;
  cam_pos_msg_.pose.pose.orientation.x = 0.0;
  cam_pos_msg_.pose.pose.orientation.y = 0.0;
  cam_pos_msg_.pose.pose.orientation.z = 0.0;

  joy_sub_ = nh.subscribe("/joy", 1, rcvJoyCallbck);
  camPos_pub_ = nh.advertise<nav_msgs::Odometry>("odom", 1);
  // camPos_pub_ = nh.advertise<nav_msgs::Odometry>("/drone0/odom", 1);
  camPos_pub_cyc_ = 1./(double)camPos_pub_rate_;
  camPos_pub_timer_ = nh.createTimer(ros::Duration(camPos_pub_cyc_),&camPosPubCallbck);

  ros::spin();
}
