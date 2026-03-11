#ifndef __PX4CTRLPARAM_H
#define __PX4CTRLPARAM_H

#include <ros/ros.h>

class Parameter_t {
public:
  struct Gain {
    double Kp0, Kp1, Kp2;
    double Kv0, Kv1, Kv2;
    double Kvi0, Kvi1, Kvi2;
    double Kvd0, Kvd1, Kvd2;
    double KAngR, KAngP, KAngY;
  };

  struct RotorDrag {
    double x, y, z;
    double k_thrust_horz;
  };

  struct MsgTimeout {
    double odom;
    double rc;
    double cmd;
    double imu;
    double bat;
    // airgrasp
    double theta;
  };

  struct ThrustMapping {
    bool print_val;
    double K1;
    double K2;
    double K3;
    bool accurate_thrust_model;
    double hover_percentage;
    bool noisy_imu;
  };

  struct RCReverse {
    bool roll;
    bool pitch;
    bool yaw;
    bool throttle;
  };

  struct AutoTakeoffLand {
    bool enable;
    bool enable_auto_arm;
    bool no_RC;
    double height;
    double speed;
  };

  struct L1 {
    double k_adapt;
    double k_filt;
    double d_max;
    double k_a_lpf;
    bool enable;
  };
  L1 l1;

  struct Thetas {
    double theta_0_min;
    double theta_0_max;
    double theta_1_min;
    double theta_1_max;
    double theta_2_min;
    double theta_2_max;
  };
  Thetas thetas;

  struct ArmCompensation {
    bool enable = false;
    double wp_x, wp_y, wp_z; // 位置误差补偿权重
    double wr_x, wr_y, wr_z; // 旋转误差补偿权重
    double alpha;            // 低通滤波系数 (0~1, 值越小滤波越强)
  };
  ArmCompensation arm_comp;

  Gain gain;
  RotorDrag rt_drag;
  MsgTimeout msg_timeout;
  RCReverse rc_reverse;
  ThrustMapping thr_map;
  AutoTakeoffLand takeoff_land;

  int pose_solver;
  double mass;
  double gra;
  double max_angle;
  double ctrl_freq_max;
  double ctrl_dt;
  double max_manual_vel;
  double low_voltage;

  bool use_bodyrate_ctrl;
  // bool print_dbg;

  Parameter_t();
  void config_from_ros_handle(const ros::NodeHandle &nh);
  void config_full_thrust(double hov);

private:
  template <typename TName, typename TVal>
  void read_essential_param(const ros::NodeHandle &nh, const TName &name,
                            TVal &val) {
    if (nh.getParam(name, val)) {
      // pass
    } else {
      ROS_ERROR_STREAM("Read param: " << name << " failed.");
      ROS_BREAK();
    }
  };
};

#endif