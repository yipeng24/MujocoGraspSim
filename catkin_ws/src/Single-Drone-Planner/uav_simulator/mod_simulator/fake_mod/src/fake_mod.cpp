#include "fake_mod/dynamics.h"
#include <iostream>
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <vector>

using namespace std;
using namespace modquad;

ros::Subscriber cmd_sub;
ros::Publisher odom_pub;
ros::Publisher imu_pub;
ros::Publisher mesh_pub;
ros::Publisher state_pub;
ros::Publisher esc_pub;
ros::Publisher esc_pub_sim;
ros::Publisher joint_state_est_pub;
ros::Publisher debug_msg_pub_;
ros::ServiceServer mode_server;
ros::Timer simulate_timer;
ros::Time get_cmdtime;

mavros_msgs::AttitudeTarget immediate_cmd;
vector<mavros_msgs::AttitudeTarget> cmd_buff;

XModQuad mod_quad;
bool rcv_cmd = false;

// airgrasp
ros::Subscriber cmd_theta_sub;
vector<sensor_msgs::JointState> cmd_theta_buff;
bool rcv_theta_cmd = false;
ros::Time get_cmd_theta_time;
sensor_msgs::JointState immediate_cmd_theta;

ros::Subscriber arm_grab_cmd_sub;
std_msgs::Bool arm_grab_cmd_msg;

void rcvCmdCallBack(const mavros_msgs::AttitudeTargetConstPtr cmd) {
  if (rcv_cmd == false) {
    rcv_cmd = true;
    cmd_buff.emplace_back(*cmd);
    get_cmdtime = ros::Time::now();
  } else {
    cmd_buff.emplace_back(*cmd);
    if ((ros::Time::now() - get_cmdtime).toSec() > mod_quad.getDelay()) {
      immediate_cmd = cmd_buff[0];
      cmd_buff.erase(cmd_buff.begin());
    }
  }
}

void rcvCmdThetaCallBack(const sensor_msgs::JointStateConstPtr cmd) {
  if (rcv_theta_cmd == false) {
    rcv_theta_cmd = true;
    cmd_theta_buff.emplace_back(*cmd);
    get_cmd_theta_time = ros::Time::now();
  } else {
    cmd_theta_buff.emplace_back(*cmd);
    if ((ros::Time::now() - get_cmd_theta_time).toSec() > mod_quad.getDelay()) {
      immediate_cmd_theta = cmd_theta_buff[0];
      cmd_theta_buff.erase(cmd_theta_buff.begin());
    }
  }
}

void rcvArmGrabCmdCallBack(const std_msgs::BoolConstPtr msg) {
  arm_grab_cmd_msg = *msg;
  ROS_INFO("[fake_mod] Received arm grab command: %s",
           arm_grab_cmd_msg.data ? "true" : "false");
}

bool rcvSetModeCallBack(mavros_msgs::SetMode::Request &req,
                        mavros_msgs::SetMode::Response &res) {
  mod_quad.setState(req.custom_mode);
  res.mode_sent = true;
  return true;
}

void simCallback(const ros::TimerEvent &e) {
  if (rcv_cmd && rcv_theta_cmd) {
    double t = mod_quad.getSimTime(); // 当前仿真时间（秒）

    Eigen::Vector3d Fw, Mb, Tau_arm;
    Fw.setZero();
    Mb.setZero();
    Tau_arm.setZero();

    // Fw << 0.0, 0.0, 0.2*cos(t+0.1);

    // Mb << 0.0, 0.05*sin(t), 0.0;

    mod_quad.setDisturbance(Fw, Mb, Tau_arm);
    mod_quad.simOneStep(immediate_cmd, immediate_cmd_theta);
  }

  odom_pub.publish(mod_quad.getOdom());
  imu_pub.publish(mod_quad.getImu());
  esc_pub.publish(mod_quad.getESC());
  mesh_pub.publish(mod_quad.getMesh());
  state_pub.publish(mod_quad.getState());
  joint_state_est_pub.publish(mod_quad.getJointState());
  debug_msg_pub_.publish(mod_quad.getSimDebug());
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "fake_mod");
  ros::NodeHandle nh("~");

  mod_quad.init(nh);

  cmd_sub = nh.subscribe("cmd", 1000, rcvCmdCallBack);

  cmd_theta_sub = nh.subscribe("cmd_theta", 1000, rcvCmdThetaCallBack);

  arm_grab_cmd_sub = nh.subscribe("/arm_grab_cmd", 10, rcvArmGrabCmdCallBack);

  odom_pub = nh.advertise<nav_msgs::Odometry>("odom", 10);
  imu_pub = nh.advertise<sensor_msgs::Imu>("imu", 10);
  mesh_pub = nh.advertise<visualization_msgs::Marker>("mesh", 10);
  state_pub = nh.advertise<mavros_msgs::State>("state", 10);
  esc_pub = nh.advertise<mavros_msgs::ESCTelemetry>("rpm", 10);
  mode_server = nh.advertiseService("set_mode", rcvSetModeCallBack);
  joint_state_est_pub =
      nh.advertise<sensor_msgs::JointState>("joint_state_est", 10);
  debug_msg_pub_ = nh.advertise<quadrotor_msgs::SimDebug>("sim_debug", 10);

  immediate_cmd.body_rate.x = 0.0;
  immediate_cmd.body_rate.y = 0.0;

  simulate_timer =
      nh.createTimer(ros::Duration(mod_quad.getTimeResolution()), simCallback);

  ros::spin();

  return 0;
}