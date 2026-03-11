#include "PX4CtrlFSM.h"
#include <ros/ros.h>
#include <signal.h>

void mySigintHandler(int sig) {
  ROS_INFO("[PX4Ctrl] exit...");
  ros::shutdown();
}

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "px4ctrl");
  ros::NodeHandle nh("~");

  signal(SIGINT, mySigintHandler);
  ros::Duration(1.0).sleep();

  Parameter_t param;
  param.config_from_ros_handle(nh);

  std::shared_ptr<clutter_hand::CH_RC_SDF> rc_sdf_ptr;
  rc_sdf_ptr = std::make_shared<clutter_hand::CH_RC_SDF>();
  rc_sdf_ptr->initMap(nh, false);

  Controller controller(param);
  controller.rc_sdf_ptr = rc_sdf_ptr;
  PX4CtrlFSM fsm(param, controller);
  fsm.rc_sdf_ptr = rc_sdf_ptr;

  ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>(
      "/mavros/state", 10,
      boost::bind(&State_Data_t::feed, &fsm.state_data, _1));

  ros::Subscriber extended_state_sub = nh.subscribe<mavros_msgs::ExtendedState>(
      "/mavros/extended_state", 10,
      boost::bind(&ExtendedState_Data_t::feed, &fsm.extended_state_data, _1));

  ros::Subscriber odom_sub = nh.subscribe<nav_msgs::Odometry>(
      "odom", 100, boost::bind(&Odom_Data_t::feed, &fsm.odom_data, _1),
      ros::VoidConstPtr(), ros::TransportHints().tcpNoDelay());

  ros::Subscriber cmd_sub = nh.subscribe<quadrotor_msgs::PositionCommand>(
      "cmd", 10, &PX4CtrlFSM::position_cmd_grab_cb, &fsm,
      ros::TransportHints().tcpNoDelay());

  ros::Subscriber imu_sub = nh.subscribe<sensor_msgs::Imu>(
      "/mavros/imu/data", // Note: do NOT change it to /mavros/imu/data_raw !!!
      100, boost::bind(&Imu_Data_t::feed, &fsm.imu_data, _1),
      ros::VoidConstPtr(), ros::TransportHints().tcpNoDelay());

  ros::Subscriber rc_sub;
  if (!param.takeoff_land.no_RC) // mavros will still publish wrong rc messages
                                 // although no RC is connected
  {
    rc_sub = nh.subscribe<mavros_msgs::RCIn>(
        "/mavros/rc/in", 10, boost::bind(&RC_Data_t::feed, &fsm.rc_data, _1));
  }

  ros::Subscriber bat_sub = nh.subscribe<sensor_msgs::BatteryState>(
      "/mavros/battery", 100,
      boost::bind(&Battery_Data_t::feed, &fsm.bat_data, _1),
      ros::VoidConstPtr(), ros::TransportHints().tcpNoDelay());

  ros::Subscriber takeoff_land_sub = nh.subscribe<quadrotor_msgs::TakeoffLand>(
      "takeoff_land", 100,
      boost::bind(&Takeoff_Land_Data_t::feed, &fsm.takeoff_land_data, _1),
      ros::VoidConstPtr(), ros::TransportHints().tcpNoDelay());

  // * airgrasp
  ros::Subscriber theta_est_sub = nh.subscribe<sensor_msgs::JointState>(
      "joint_state_est", 100, &PX4CtrlFSM::joint_state_est_cb, &fsm,
      ros::TransportHints().tcpNoDelay());

  ros::Subscriber theta_cmd_manual_sub = nh.subscribe<sensor_msgs::JointState>(
      "joint_state_cmd_manual", 100,
      boost::bind(&Joint_State_Data_t::feed, &fsm.theta_cmd_manual_data, _1),
      ros::VoidConstPtr(), ros::TransportHints().tcpNoDelay());

  ros::Subscriber joy_sub =
      nh.subscribe<sensor_msgs::Joy>("joy_cmd", 100, &PX4CtrlFSM::joy_grab_cb,
                                     &fsm, ros::TransportHints().tcpNoDelay());
  fsm.theta_cmd_pub =
      nh.advertise<sensor_msgs::JointState>("joint_state_cmd", 10);
  fsm.end_pose_est_pub = nh.advertise<nav_msgs::Odometry>("end_pose_est", 10);
  fsm.end_pose_ref_pub = nh.advertise<nav_msgs::Odometry>("end_pose_ref", 10);
  fsm.grab_cmd_pub = nh.advertise<std_msgs::Bool>("/arm_grab_cmd", 10);

  fsm.ctrl_FCU_pub = nh.advertise<mavros_msgs::AttitudeTarget>(
      "/mavros/setpoint_raw/attitude", 10);
  fsm.traj_start_trigger_pub =
      nh.advertise<geometry_msgs::PoseStamped>("/traj_start_trigger", 10);

  fsm.debug_pub =
      nh.advertise<quadrotor_msgs::Px4ctrlDebug>("/debugPx4ctrl", 10);

  fsm.set_FCU_mode_srv =
      nh.serviceClient<mavros_msgs::SetMode>("/mavros/set_mode");
  fsm.arming_client_srv =
      nh.serviceClient<mavros_msgs::CommandBool>("/mavros/cmd/arming");
  fsm.reboot_FCU_srv =
      nh.serviceClient<mavros_msgs::CommandLong>("/mavros/cmd/command");

  if (param.takeoff_land.no_RC) {
    ROS_WARN("PX4CTRL] Remote controller disabled, be careful!");
  } else {
    ROS_INFO("PX4CTRL] Waiting for RC");
    while (ros::ok()) {
      ros::spinOnce();
      if (fsm.rc_is_received(ros::Time::now())) {
        ROS_INFO("[PX4CTRL] RC received.");
        break;
      }
      ros::Duration(0.1).sleep();
    }
  }

  int trials = 0;
  while (ros::ok() && !fsm.state_data.current_state.connected) {
    ros::spinOnce();
    ros::Duration(1.0).sleep();
    if (trials++ > 5)
      ROS_ERROR("Unable to connnect to PX4!!!");
  }

  ros::Rate r(param.ctrl_freq_max);
  while (ros::ok()) {
    r.sleep();
    ros::spinOnce();
    fsm.process(); // We DO NOT rely on feedback as trigger, since there is no
                   // significant performance difference through our test.
  }

  return 0;
}
