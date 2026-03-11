#ifndef __PX4CTRLFSM_H
#define __PX4CTRLFSM_H

#include <ros/assert.h>
#include <ros/ros.h>

#include <geometry_msgs/PoseStamped.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/CommandLong.h>
#include <mavros_msgs/SetMode.h>
#include <nav_msgs/Odometry.h>

#include "input.h"
// #include "ThrustCurve.h"
#include "controller.h"
#include <ch_rc_sdf/ch_rc_sdf.h>

struct AutoTakeoffLand_t {
  bool landed{true};
  ros::Time toggle_takeoff_land_time;
  std::pair<bool, ros::Time> delay_trigger{
      std::pair<bool, ros::Time>(false, ros::Time(0))};
  Eigen::Vector4d start_pose;

  static constexpr double MOTORS_SPEEDUP_TIME =
      3.0; // motors idle running for 3 seconds before takeoff
  static constexpr double DELAY_TRIGGER_TIME =
      2.0; // Time to be delayed when reach at target height
};

class PX4CtrlFSM {
public:
  Parameter_t &param;

  RC_Data_t rc_data;
  State_Data_t state_data;
  ExtendedState_Data_t extended_state_data;
  Odom_Data_t odom_data;
  Imu_Data_t imu_data;
  Command_Data_t cmd_data;
  Battery_Data_t bat_data;
  Takeoff_Land_Data_t takeoff_land_data;

  // * airgrasp
  Joint_State_Data_t theta_cur_data, theta_cmd_manual_data;
  ros::Publisher theta_cmd_pub;
  Joy_Data_t joy_cmd_data;
  std::shared_ptr<clutter_hand::CH_RC_SDF> rc_sdf_ptr;

  Controller &controller;

  ros::Publisher traj_start_trigger_pub;
  ros::Publisher ctrl_FCU_pub;
  ros::Publisher debug_pub; // debug
  ros::ServiceClient set_FCU_mode_srv;
  ros::ServiceClient arming_client_srv;
  ros::ServiceClient reboot_FCU_srv;

  quadrotor_msgs::Px4ctrlDebug debug_msg; // debug

  Eigen::Vector4d hover_pose;
  ros::Time last_set_hover_pose_time;

  // airgrasp
  Eigen::Vector3d hover_theta;
  ros::Publisher joint_cmd_pub;

  enum State_t {
    MANUAL_CTRL = 1, // px4ctrl is deactived. FCU is controled by the remote
                     // controller only
    AUTO_HOVER, // px4ctrl is actived, it will keep the drone hover from odom
                // measurments while waiting for commands from PositionCommand
                // topic.
    CMD_CTRL,   // px4ctrl is actived, and controling the drone.
    AUTO_TAKEOFF,
    AUTO_LAND
  };

  PX4CtrlFSM(Parameter_t &, Controller &);
  void process();
  bool rc_is_received(const ros::Time &now_time);
  bool cmd_is_received(const ros::Time &now_time);
  bool odom_is_received(const ros::Time &now_time);
  bool imu_is_received(const ros::Time &now_time);
  bool bat_is_received(const ros::Time &now_time);
  bool recv_new_odom();
  State_t get_state() { return state; }
  bool get_landed() { return takeoff_land.landed; }

  // airgrasp
  bool theta_fb_is_received(const ros::Time &now_time);
  bool joy_cmd_is_received(const ros::Time &now_time);
  void joint_state_est_cb(const sensor_msgs::JointState::ConstPtr &msg);

  // Grab command queue and callbacks
  std::queue<uint8_t> grab_cmd_queue;
  void
  position_cmd_grab_cb(const quadrotor_msgs::PositionCommandConstPtr &pMsg);
  void joy_grab_cb(const sensor_msgs::JoyConstPtr &pMsg);
  void pub_grab_cmd();

  ros::Publisher end_pose_est_pub;
  ros::Publisher end_pose_ref_pub;
  ros::Publisher grab_cmd_pub;

private:
  State_t state; // Should only be changed in PX4CtrlFSM::process() function!
  AutoTakeoffLand_t takeoff_land;

  // ---- control related ----
  Desired_State_t get_hover_des();
  Desired_State_t get_cmd_des();

  // ---- auto takeoff/land ----
  void motors_idling(const Imu_Data_t &imu, Controller_Output_t &u);
  void land_detector(const State_t state, const Desired_State_t &des,
                     const Odom_Data_t &odom); // Detect landing
  void set_start_pose_for_takeoff_land(const Odom_Data_t &odom);
  Desired_State_t get_rotor_speed_up_des(const ros::Time now);
  Desired_State_t get_takeoff_land_des(const double speed);

  // ---- tools ----
  void set_hov_with_odom();
  void set_hov_with_rc();

  bool toggle_offboard_mode(
      bool on_off); // It will only try to toggle once, so not blocked.
  bool toggle_arm_disarm(
      bool arm); // It will only try to toggle once, so not blocked.
  void reboot_FCU();

  void publish_bodyrate_ctrl(const Controller_Output_t &u,
                             const ros::Time &stamp);
  void publish_attitude_ctrl(const Controller_Output_t &u,
                             const ros::Time &stamp);
  void publish_trigger(const nav_msgs::Odometry &odom_msg);

  // airgrasp
  void set_theta_manul_des(Desired_State_t des);
  void thetasLimit(Eigen::Vector3d &thetas);
};

#endif