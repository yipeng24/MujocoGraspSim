#include "node/data_callback_node.h"

// #ifdef ROS
using rot_util = rotation_util::RotUtil;

struct joyState{
  double X = 0.0;
  double Y = 0.0;
  double Z = 0.0;
  double yaw = 0.0;
};

void DataCallBacks::init_ros(ros::NodeHandle& nh){

    odom_sub_ = nh.subscribe<nav_msgs::Odometry>("odom", 10, &DataCallBacks::odom_callback, this, ros::TransportHints().tcpNoDelay());
    plan_trigger_sub_ = nh.subscribe<geometry_msgs::PoseStamped>("triger", 10, &DataCallBacks::triger_plan_callback, this, ros::TransportHints().tcpNoDelay());
    land_triger_sub_ = nh.subscribe<geometry_msgs::PoseStamped>("land_triger", 10, &DataCallBacks::land_triger_callback, this, ros::TransportHints().tcpNoDelay());
    target_sub_ = nh.subscribe<nav_msgs::Odometry>("target", 10, &DataCallBacks::target_callback, this, ros::TransportHints().tcpNoDelay());
    goal_sub_ = nh.subscribe<geometry_msgs::PoseStamped>("goal", 10, &DataCallBacks::goal_callback, this, ros::TransportHints().tcpNoDelay());

    goal_full_state_sub_ = nh.subscribe<quadrotor_msgs::UAMFullState>("uam_state_goal_cmd", 10, &DataCallBacks::goal_full_state_callback, this, ros::TransportHints().tcpNoDelay());


    // gridmap_sub_ = nh.subscribe<quadrotor_msgs::OccMap3d>("gridmap_inflate", 1, &DataCallBacks::gridmap_callback, this, ros::TransportHints().tcpNoDelay());

    gridmap_sub_ = nh.subscribe<sensor_msgs::PointCloud2>("/sdf_map/esdf", 1, &DataCallBacks::gridmap_callback, this, ros::TransportHints().tcpNoDelay());
    
    joy_sub_ = nh.subscribe<sensor_msgs::Joy>("joy", 10, &DataCallBacks::joy_callback, this);
}

  void DataCallBacks::joy_callback(const sensor_msgs::Joy::ConstPtr& msgPtr){
    joyState joy_state;
    joy_state.X = msgPtr->axes.at(3);
    joy_state.Y = msgPtr->axes.at(4);
    joy_state.Z = msgPtr->axes.at(1);
    joy_state.yaw = msgPtr->axes.at(0);

    GData<double> tracking_ang_data;
    if (dataManagerPtr_->get_data(dataManagerPtr_->tracking_angle_info_, tracking_ang_data)){
      tracking_ang_data.data_ += joy_state.X*1/180*M_PI;
      rot_util::rad_limit(tracking_ang_data.data_);
      dataManagerPtr_->write_data(tracking_ang_data, dataManagerPtr_->tracking_angle_info_);
      INFO_MSG("tracking_ang: " << tracking_ang_data.data_);
    }

    GData<double> tracking_dis_data;
    if (dataManagerPtr_->get_data(dataManagerPtr_->tracking_dis_info_, tracking_dis_data)){
      tracking_dis_data.data_ += joy_state.Y * 0.2;
      dataManagerPtr_->write_data(tracking_dis_data, dataManagerPtr_->tracking_dis_info_);
    }


    GData<double> tracking_h_data;
    if (dataManagerPtr_->get_data(dataManagerPtr_->tracking_height_info_, tracking_h_data)){
      tracking_h_data.data_ += joy_state.Z * 0.1;
      dataManagerPtr_->write_data(tracking_h_data, dataManagerPtr_->tracking_height_info_);
    }
  }


  void DataCallBacks::odom_callback(const nav_msgs::Odometry::ConstPtr& msgPtr) {
    Odom odom_data;
    odom_data.odom_p_ << msgPtr->pose.pose.position.x, msgPtr->pose.pose.position.y, msgPtr->pose.pose.position.z;
    odom_data.odom_v_ << msgPtr->twist.twist.linear.x, msgPtr->twist.twist.linear.y, msgPtr->twist.twist.linear.z;
    odom_data.odom_q_.w() = msgPtr->pose.pose.orientation.w;
    odom_data.odom_q_.x() = msgPtr->pose.pose.orientation.x;
    odom_data.odom_q_.y() = msgPtr->pose.pose.orientation.y;
    odom_data.odom_q_.z() = msgPtr->pose.pose.orientation.z;
    dataManagerPtr_->write_odom(odom_data, dataManagerPtr_->odom_info_);
  }

  void DataCallBacks::triger_plan_callback(const geometry_msgs::PoseStampedConstPtr& msgPtr) {
    Odom goal_data;
    goal_data.odom_p_ << msgPtr->pose.position.x, msgPtr->pose.position.y, msgPtr->pose.position.z;
    INFO_MSG_GREEN("[data_callback] Plan Trigger!");
    // dataManagerPtr_->write_odom(goal_data, dataManagerPtr_->goal_info_);
    dataManagerPtr_->auto_mode_ = true;
    // NOTE: do NOT set plan_trigger_received_ here.
    // Setting it here causes FSM to immediately jump HOVER→GOAL with no valid goal,
    // which makes the planner plan to (0,0,0) and send erratic position commands.
    // plan_trigger_received_ is set only when an explicit goal arrives (goal_callback /
    // goal_full_state_callback), so the FSM stays in HOVER until the user sends a goal.
    // dataManagerPtr_->plan_trigger_received_ = true;
  }

  void DataCallBacks::land_triger_callback(const geometry_msgs::PoseStampedConstPtr& msgPtr) {
    INFO_MSG_GREEN("[data_callback] Land Trigger!");
    dataManagerPtr_->land_trigger_received_ = true;
  }

  void DataCallBacks::target_callback(const nav_msgs::Odometry::ConstPtr& msgPtr) {
    static TimePoint last_stamp;
    static bool has_last_ = false;
    if (!has_last_ || durationSecond(TimeNow(), last_stamp) >= 0.09){
      last_stamp = TimeNow();
      has_last_ = true;
    }else{
      has_last_ = true;
      return;
    }

    Eigen::Vector3d target_p;
    target_p << msgPtr->pose.pose.position.x, msgPtr->pose.pose.position.y, msgPtr->pose.pose.position.z;
    Eigen::Quaterniond target_q;
    target_q.w() = msgPtr->pose.pose.orientation.w;
    target_q.x() = msgPtr->pose.pose.orientation.x;
    target_q.y() = msgPtr->pose.pose.orientation.y;
    target_q.z() = msgPtr->pose.pose.orientation.z;
    double target_v = sqrt(msgPtr->twist.twist.linear.x *msgPtr->twist.twist.linear.x + msgPtr->twist.twist.linear.y * msgPtr->twist.twist.linear.y);
    dataManagerPtr_->car_ekf_ptr_->update_p_state_diff_v(target_p, Eigen::Vector3d::Zero(), TimeNow());
  }

  void DataCallBacks::goal_callback(const geometry_msgs::PoseStampedConstPtr& msgPtr){
    dataManagerPtr_->auto_mode_ = true;
    dataManagerPtr_->plan_trigger_received_ = true;

    Odom goal;
    goal.odom_p_ << msgPtr->pose.position.x, msgPtr->pose.position.y, msgPtr->pose.position.z;
    // Use z from PoseStamped; fall back to 1.5m if z is near zero (2D nav goal)
    if (goal.odom_p_.z() < 0.1) goal.odom_p_.z() = 1.5;
    goal.odom_v_.setZero();
    goal.odom_a_.setZero();
    goal.odom_q_.w() = msgPtr->pose.orientation.w;
    goal.odom_q_.x() = msgPtr->pose.orientation.x;
    goal.odom_q_.y() = msgPtr->pose.orientation.y;
    goal.odom_q_.z() = msgPtr->pose.orientation.z;

    goal.odom_time_stamp_ms_ = TimeNow();
    dataManagerPtr_->write_odom(goal, dataManagerPtr_->goal_info_);
  }

  void DataCallBacks::goal_full_state_callback(const quadrotor_msgs::UAMFullState::ConstPtr& msgPtr){
    std::cout << "receive goal full state callback" << std::endl;
    dataManagerPtr_->auto_mode_ = true;
    dataManagerPtr_->plan_trigger_received_ = true;

    Odom goal;
    goal.odom_p_ << msgPtr->pose.position.x, 
                    msgPtr->pose.position.y, 
                    msgPtr->pose.position.z;
    goal.odom_v_.setZero();
    goal.odom_a_.setZero();

    // 姿态（四元数，归一化以防止累计误差）
    Eigen::Quaterniond q(msgPtr->pose.orientation.w,
                         msgPtr->pose.orientation.x,
                         msgPtr->pose.orientation.y,
                         msgPtr->pose.orientation.z);
    goal.odom_q_ = q.normalized();


    // 关节角 & 角速度（float64[] -> std::vector<double> -> Eigen::VectorXd）
    // 使用 Map 构造再赋值，会**复制**到 goal.theta_ / goal.dtheta_ 中，生命周期安全
    if (!msgPtr->theta.empty()) {
      size_t n = std::min<size_t>(3, msgPtr->theta.size());
      goal.theta_ = Eigen::VectorXd::Map(msgPtr->theta.data(),
                                          static_cast<Eigen::Index>(n));
    } else {
        goal.theta_.resize(0);
    }

    if (!msgPtr->dtheta.empty()) {
      size_t n = std::min<size_t>(3, msgPtr->dtheta.size());
      goal.dtheta_ = Eigen::VectorXd::Map(msgPtr->dtheta.data(),
                                          static_cast<Eigen::Index>(n));
    } else {
        goal.dtheta_.resize(0);
    }

    // （可选）一致性检查：维度不一致时给个告警
    if (goal.dtheta_.size() > 0 && goal.theta_.size() != goal.dtheta_.size()) {
        ROS_WARN_STREAM_THROTTLE(1.0,
            "[goal_full_state_callback] thetas (" << goal.theta_.size()
            << ") and dthetas (" << goal.dtheta_.size() << ") have different sizes.");
    }

    goal.odom_time_stamp_ms_ = TimeNow();
    dataManagerPtr_->write_odom(goal, dataManagerPtr_->goal_info_);
  }

  // void DataCallBacks::gridmap_callback(const quadrotor_msgs::OccMap3dConstPtr& msgPtr) {
  //   std::unique_lock<std::mutex> lck(dataManagerPtr_->map_mutex_);
  //   dataManagerPtr_->map_msg_ =  *msgPtr;
  //   dataManagerPtr_->map_received_ = true;
  //   // ROS_ERROR("receive the map");
  // }

  void DataCallBacks::gridmap_callback(const sensor_msgs::PointCloud2ConstPtr& msgPtr) {
    std::unique_lock<std::mutex> lck(dataManagerPtr_->map_mutex_);
    // dataManagerPtr_->map_msg_ =  *msgPtr;
    dataManagerPtr_->map_received_ = true;
    // ROS_ERROR("receive the map");
  }


// #endif