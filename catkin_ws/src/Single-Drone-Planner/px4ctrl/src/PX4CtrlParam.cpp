#include "PX4CtrlParam.h"

Parameter_t::Parameter_t() {}

void Parameter_t::config_from_ros_handle(const ros::NodeHandle &nh) {
  read_essential_param(nh, "gain/Kp0", gain.Kp0);
  read_essential_param(nh, "gain/Kp1", gain.Kp1);
  read_essential_param(nh, "gain/Kp2", gain.Kp2);
  read_essential_param(nh, "gain/Kv0", gain.Kv0);
  read_essential_param(nh, "gain/Kv1", gain.Kv1);
  read_essential_param(nh, "gain/Kv2", gain.Kv2);
  read_essential_param(nh, "gain/Kvi0", gain.Kvi0);
  read_essential_param(nh, "gain/Kvi1", gain.Kvi1);
  read_essential_param(nh, "gain/Kvi2", gain.Kvi2);
  read_essential_param(nh, "gain/KAngR", gain.KAngR);
  read_essential_param(nh, "gain/KAngP", gain.KAngP);
  read_essential_param(nh, "gain/KAngY", gain.KAngY);

  read_essential_param(nh, "rotor_drag/x", rt_drag.x);
  read_essential_param(nh, "rotor_drag/y", rt_drag.y);
  read_essential_param(nh, "rotor_drag/z", rt_drag.z);
  read_essential_param(nh, "rotor_drag/k_thrust_horz", rt_drag.k_thrust_horz);

  read_essential_param(nh, "msg_timeout/odom", msg_timeout.odom);
  read_essential_param(nh, "msg_timeout/rc", msg_timeout.rc);
  read_essential_param(nh, "msg_timeout/cmd", msg_timeout.cmd);
  read_essential_param(nh, "msg_timeout/imu", msg_timeout.imu);
  read_essential_param(nh, "msg_timeout/bat", msg_timeout.bat);

  // airgrasp
  read_essential_param(nh, "msg_timeout/theta", msg_timeout.theta);

  read_essential_param(nh, "pose_solver", pose_solver);
  read_essential_param(nh, "mass", mass);
  read_essential_param(nh, "gra", gra);
  read_essential_param(nh, "ctrl_freq_max", ctrl_freq_max);
  ctrl_dt = 1.0 / ctrl_freq_max;
  read_essential_param(nh, "use_bodyrate_ctrl", use_bodyrate_ctrl);
  read_essential_param(nh, "max_manual_vel", max_manual_vel);
  read_essential_param(nh, "max_angle", max_angle);
  read_essential_param(nh, "low_voltage", low_voltage);

  read_essential_param(nh, "rc_reverse/roll", rc_reverse.roll);
  read_essential_param(nh, "rc_reverse/pitch", rc_reverse.pitch);
  read_essential_param(nh, "rc_reverse/yaw", rc_reverse.yaw);
  read_essential_param(nh, "rc_reverse/throttle", rc_reverse.throttle);

  read_essential_param(nh, "auto_takeoff_land/enable", takeoff_land.enable);
  read_essential_param(nh, "auto_takeoff_land/enable_auto_arm",
                       takeoff_land.enable_auto_arm);
  read_essential_param(nh, "auto_takeoff_land/no_RC", takeoff_land.no_RC);
  read_essential_param(nh, "auto_takeoff_land/takeoff_height",
                       takeoff_land.height);
  read_essential_param(nh, "auto_takeoff_land/takeoff_land_speed",
                       takeoff_land.speed);

  read_essential_param(nh, "thrust_model/print_value", thr_map.print_val);
  read_essential_param(nh, "thrust_model/K1", thr_map.K1);
  read_essential_param(nh, "thrust_model/K2", thr_map.K2);
  read_essential_param(nh, "thrust_model/K3", thr_map.K3);
  read_essential_param(nh, "thrust_model/accurate_thrust_model",
                       thr_map.accurate_thrust_model);
  read_essential_param(nh, "thrust_model/hover_percentage",
                       thr_map.hover_percentage);
  read_essential_param(nh, "thrust_model/noisy_imu", thr_map.noisy_imu);

  // airgrasp
  read_essential_param(nh, "l1/k_adapt", l1.k_adapt);
  read_essential_param(nh, "l1/k_filt", l1.k_filt);
  read_essential_param(nh, "l1/d_max", l1.d_max);
  read_essential_param(nh, "l1/enable", l1.enable);
  read_essential_param(nh, "l1/k_a_lpf", l1.k_a_lpf);

  read_essential_param(nh, "thetas/theta_0_min", thetas.theta_0_min);
  read_essential_param(nh, "thetas/theta_0_max", thetas.theta_0_max);
  read_essential_param(nh, "thetas/theta_1_min", thetas.theta_1_min);
  read_essential_param(nh, "thetas/theta_1_max", thetas.theta_1_max);
  read_essential_param(nh, "thetas/theta_2_min", thetas.theta_2_min);
  read_essential_param(nh, "thetas/theta_2_max", thetas.theta_2_max);

  read_essential_param(nh, "arm_compensation/enable", arm_comp.enable);
  std::cout << "arm_comp.enable: " << arm_comp.enable << std::endl;
  read_essential_param(nh, "arm_compensation/wp_x", arm_comp.wp_x);
  read_essential_param(nh, "arm_compensation/wp_y", arm_comp.wp_y);
  read_essential_param(nh, "arm_compensation/wp_z", arm_comp.wp_z);
  read_essential_param(nh, "arm_compensation/wr_x", arm_comp.wr_x);
  read_essential_param(nh, "arm_compensation/wr_y", arm_comp.wr_y);
  read_essential_param(nh, "arm_compensation/wr_z", arm_comp.wr_z);
  read_essential_param(nh, "arm_compensation/alpha", arm_comp.alpha);

  max_angle /= (180.0 / M_PI);

  if (takeoff_land.enable_auto_arm && !takeoff_land.enable) {
    takeoff_land.enable_auto_arm = false;
    ROS_ERROR("\"enable_auto_arm\" is only allowd with \"auto_takeoff_land\" "
              "enabled.");
  }
  if (takeoff_land.no_RC &&
      (!takeoff_land.enable_auto_arm || !takeoff_land.enable)) {
    takeoff_land.no_RC = false;
    ROS_ERROR("\"no_RC\" is only allowd with both \"auto_takeoff_land\" and "
              "\"enable_auto_arm\" enabled.");
  }

  if (thr_map.print_val) {
    ROS_WARN("You should disable \"print_value\" if you are in regular usage.");
  }
};

// void Parameter_t::config_full_thrust(double hov)
// {
// 	full_thrust = mass * gra / hov;
// };