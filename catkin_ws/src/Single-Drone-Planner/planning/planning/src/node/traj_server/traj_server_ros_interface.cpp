#include "node/traj_server_node.h"

using rot_util = rotation_util::RotUtil;

static std::vector<ros::Publisher> pubs;

void TrajServer::init_ros(ros::NodeHandle& nh){
  nh_ = nh;
  pos_cmd_pub_ = nh.advertise<quadrotor_msgs::PositionCommand>("position_cmd", 50);
  stop_pub_ = nh.advertise<std_msgs::Empty>("stop_signal", 10);

  // for (int i = 0; i < 100; i++){
  //   std::string topic_name = "/traj/" + std::to_string(i);
  //   ros::Publisher pub = nh_.advertise<quadrotor_msgs::PositionCommand>(topic_name, 50);
  //   pubs.push_back(pub);
  // }


}

bool TrajServer::stop_propeller(){
  stop_pub_.publish(std_msgs::Empty());
  return true;
}

void TrajServer::publish_cmd(int traj_id,
                             const Eigen::Vector3d& p,
                             const Eigen::Vector3d& v,
                             const Eigen::Vector3d& a,
                             const Eigen::Vector3d& j,
                             const int& grasp_cmd,
                             double& y, double yd, bool look_forward,
                             const Eigen::Vector3d& odom_p, const Eigen::Vector3d& odom_v, const Eigen::Vector3d& odom_a,
                             const double& odom_yaw, TimePoint& odom_timestamp_ms, TimePoint& sample_time)
{
  quadrotor_msgs::PositionCommand cmd;
  cmd.header.stamp = ros::Time::now();
  cmd.header.stamp.nsec -= (uint32_t)(0.02 * 1e9);
  cmd.header.frame_id = "world";
  cmd.trajectory_flag = quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_READY;
  cmd.trajectory_id = traj_id;

  cmd.position.x = p.x();
  cmd.position.y = p.y();
  cmd.position.z = p.z();
  cmd.velocity.x = v.x();
  cmd.velocity.y = v.y();
  cmd.velocity.z = v.z();
  cmd.acceleration.x = a.x();
  cmd.acceleration.y = a.y();
  cmd.acceleration.z = a.z();
  cmd.jerk.x = j.x();
  cmd.jerk.y = j.y();
  cmd.jerk.z = j.z();
  cmd.grab_cmd = grasp_cmd;

  if (look_forward)
  {
    double yaw_exp_w = atan2(v(1), v(0));
    double yaw_exp_w_limit = rot_util::truncate_error_angle(odom_yaw, yaw_exp_w, 0.15);

    if (v.head(2).norm() < 0.5){
      yaw_exp_w_limit = y;
    }else{
      y = yaw_exp_w_limit;
    }
    cmd.yaw = yaw_exp_w_limit;
    cmd.yaw_dot = 0.0;
  }
  else
  {
    cmd.yaw = y;
    cmd.yaw_dot = yd;
  }

  if((p-last_pos_cmd1_).norm() > 1.0 ){
      INFO_MSG_RED("[pub_cmd] Error! pos jump, please adjust the plan_hz and plan_estimated_time.");
      INFO_MSG_RED("last_pos_cmd_:" << last_pos_cmd1_.transpose() << " p:" << p.transpose());
  }
  last_pos_cmd1_ = p;
  last_vel_cmd1_ = v;
  last_acc_cmd1_ = a;


  pos_cmd_pub_.publish(cmd);
}


void TrajServer::publish_cmd(int traj_id,
                             const Eigen::Vector3d& p,
                             const Eigen::Vector3d& v,
                             const Eigen::Vector3d& a,
                             const Eigen::Vector3d& j,
                             const Eigen::VectorXd& theta, 
                             const Eigen::VectorXd& dtheta,
                             const int& grasp_cmd,
                             double& y, double yd, bool look_forward,
                             const Eigen::Vector3d& odom_p, const Eigen::Vector3d& odom_v, const Eigen::Vector3d& odom_a,
                             const double& odom_yaw, TimePoint& odom_timestamp_ms, TimePoint& sample_time)
{
  quadrotor_msgs::PositionCommand cmd;
  cmd.header.stamp = ros::Time::now();
  cmd.header.stamp.nsec -= (uint32_t)(0.02 * 1e9);
  cmd.header.frame_id = "world";
  cmd.trajectory_flag = quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_READY;
  cmd.trajectory_id = traj_id;

  cmd.position.x = p.x();
  cmd.position.y = p.y();
  cmd.position.z = p.z();
  cmd.velocity.x = v.x();
  cmd.velocity.y = v.y();
  cmd.velocity.z = v.z();
  cmd.acceleration.x = a.x();
  cmd.acceleration.y = a.y();
  cmd.acceleration.z = a.z();
  cmd.jerk.x = j.x();
  cmd.jerk.y = j.y();
  cmd.jerk.z = j.z();
  cmd.grab_cmd = grasp_cmd;

  if (look_forward)
  {
    double yaw_exp_w = atan2(v(1), v(0));
    double yaw_exp_w_limit = rot_util::truncate_error_angle(odom_yaw, yaw_exp_w, 0.15);

    if (v.head(2).norm() < 0.5){
      yaw_exp_w_limit = y;
    }else{
      y = yaw_exp_w_limit;
    }
    cmd.yaw = yaw_exp_w_limit;
    cmd.yaw_dot = 0.0;
  }
  else
  {
    cmd.yaw = y;
    cmd.yaw_dot = yd;
  }

  if((p-last_pos_cmd1_).norm() > 1.0 ){
      INFO_MSG_RED("[pub_cmd] Error! pos jump, please adjust the plan_hz and plan_estimated_time.");
      INFO_MSG_RED("last_pos_cmd_:" << last_pos_cmd1_.transpose() << " p:" << p.transpose());
  }
  last_pos_cmd1_ = p;
  last_vel_cmd1_ = v;
  last_acc_cmd1_ = a;

  sensor_msgs::JointState theta_cmd;
  theta_cmd.name.push_back("arm_joint_pitch");
  theta_cmd.name.push_back("arm_joint_pitch2");
  theta_cmd.name.push_back("arm_joint_roll");
  theta_cmd.position.push_back(theta(0));
  theta_cmd.position.push_back(theta(1));
  theta_cmd.position.push_back(theta(2));
  theta_cmd.velocity.push_back(dtheta(0));
  theta_cmd.velocity.push_back(dtheta(1));
  theta_cmd.velocity.push_back(dtheta(2));
  cmd.theta = theta_cmd;

  last_theta_cmd_ = theta;
  last_dtheta_cmd_ = dtheta;

  pos_cmd_pub_.publish(cmd);
}

void TrajServer::traj2vis(const TrajData& traj_data){
  std::string topic_name = "/traj/" + std::to_string(traj_data.traj_id_);
  INFO_MSG_RED(topic_name);
  // pos_cmd_pub = nh_.advertise<quadrotor_msgs::PositionCommand>(topic_name, 200);

  for (double t = 0; t < traj_data.getTotalDuration(); t += 0.02){
    Eigen::Vector3d p, v, a, j;
    double yaw, dyaw;
    p = traj_data.getPos(t);
    v = traj_data.getVel(t);
    a = traj_data.getAcc(t);
    j = traj_data.getJer(t);
    yaw = traj_data.getAngle(t);
    dyaw = traj_data.getAngleRate(t);

    quadrotor_msgs::PositionCommand cmd;
    double dt = durationSecond(traj_data.start_time_, TimeNow());
    ros::Time stamp = ros::Time::now() + ros::Duration(dt + t);
    cmd.header.stamp = stamp;
    cmd.header.frame_id = "world";
    cmd.trajectory_flag = quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_READY;
    cmd.trajectory_id = traj_data.traj_id_;

    cmd.position.x = p.x();
    cmd.position.y = p.y();
    cmd.position.z = p.z();
    cmd.velocity.x = v.x();
    cmd.velocity.y = v.y();
    cmd.velocity.z = v.z();
    cmd.acceleration.x = a.x();
    cmd.acceleration.y = a.y();
    cmd.acceleration.z = a.z();    
    cmd.jerk.x = j.x();
    cmd.jerk.y = j.y();
    cmd.jerk.z = j.z();

    cmd.yaw = yaw;
    cmd.yaw_dot = dyaw;


    // pubs[int(traj_data.traj_id_%100)].publish(cmd);
  }

  double t = traj_data.getTotalDuration();
  Eigen::Vector3d p, v, a, j;
  double yaw, dyaw;
  p = traj_data.getPos(t);
  v = traj_data.getVel(t);
  a = traj_data.getAcc(t);
  j = traj_data.getJer(t);
  yaw = traj_data.getAngle(t);
  dyaw = traj_data.getAngleRate(t);

  quadrotor_msgs::PositionCommand cmd;
  double dt = durationSecond(traj_data.start_time_, TimeNow());
  ros::Time stamp = ros::Time::now() + ros::Duration(dt + t);
  cmd.header.stamp = stamp;
  cmd.header.frame_id = "world";
  cmd.trajectory_flag = quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_READY;
  cmd.trajectory_id = traj_data.traj_id_;

  cmd.position.x = p.x();
  cmd.position.y = p.y();
  cmd.position.z = p.z();
  cmd.velocity.x = v.x();
  cmd.velocity.y = v.y();
  cmd.velocity.z = v.z();
  cmd.acceleration.x = a.x();
  cmd.acceleration.y = a.y();
  cmd.acceleration.z = a.z();    
  cmd.jerk.x = j.x();
  cmd.jerk.y = j.y();
  cmd.jerk.z = j.z();

  cmd.yaw = yaw;
  cmd.yaw_dot = dyaw;


  // pubs[int(traj_data.traj_id_%100)].publish(cmd);

}
