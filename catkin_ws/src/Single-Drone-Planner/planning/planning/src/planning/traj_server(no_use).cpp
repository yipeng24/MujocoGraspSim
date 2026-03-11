#include <nav_msgs/Odometry.h>
#include <quadrotor_msgs/PolyTraj.h>
#include <quadrotor_msgs/PositionCommand.h>
#include <ros/ros.h>
#include <std_msgs/Empty.h>
#include <visualization_msgs/Marker.h>

#include <traj_opt/trajectory.hpp>

ros::Publisher pos_cmd_pub_;
ros::Time heartbeat_time_;
bool receive_traj_ = false;
bool flight_start_ = false;
quadrotor_msgs::PolyTraj trajMsg_, trajMsg_last_;
Eigen::Vector3d last_p_;
double last_yaw_ = 0;
int last_id_ = 0;

void publish_cmd(int traj_id,
                 const Eigen::Vector3d &p,
                 const Eigen::Vector3d &v,
                 const Eigen::Vector3d &a,
                 double y, double yd) {
  quadrotor_msgs::PositionCommand cmd;
  cmd.header.stamp = ros::Time::now();
  cmd.header.frame_id = "world";
  cmd.trajectory_flag = quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_READY;
  cmd.trajectory_id = traj_id;

  cmd.position.x = p(0);
  cmd.position.y = p(1);
  cmd.position.z = p(2);
  cmd.velocity.x = v(0);
  cmd.velocity.y = v(1);
  cmd.velocity.z = v(2);
  cmd.acceleration.x = a(0);
  cmd.acceleration.y = a(1);
  cmd.acceleration.z = a(2);
  cmd.yaw = y;
  cmd.yaw_dot = yd;
  pos_cmd_pub_.publish(cmd);
  last_p_ = p;
}

bool exe_traj(const quadrotor_msgs::PolyTraj &trajMsg) {
  double t = (ros::Time::now() - trajMsg.start_time).toSec();
  if (t > 0) {

    if (trajMsg.hover) {
      if (trajMsg.hover_p.size() != 3) {
        ROS_ERROR("[traj_server] hover_p is not 3d!");
      }
      Eigen::Vector3d p, v0;
      p.x() = trajMsg.hover_p[0];
      p.y() = trajMsg.hover_p[1];
      p.z() = trajMsg.hover_p[2];
      v0.setZero();
      publish_cmd(trajMsg.traj_id, p, v0, v0, last_yaw_, 0);  // TODO yaw
      return true;
    }
    // if (trajMsg.order != 7) {
    //   ROS_ERROR("[traj_server] Only support trajectory order equals 7 now!");
    //   return false;
    // }
    if (trajMsg.duration.size() * (trajMsg.order + 1) != trajMsg.coef_x.size()) {
      ROS_ERROR("[traj_server] WRONG trajectory parameters!");
      return false;
    }
    int piece_nums = trajMsg.duration.size();
    std::vector<double> dura(piece_nums);
    Eigen::Vector3d p, v, a, j;
    double yaw;
    double dyaw = 0.0;
    if (trajMsg.order == 7){
      std::vector<Piece<7>::CoefficientMat> cMats(piece_nums);
      std::vector<AngleCoefficientMat> cMats_angle(piece_nums);
      for (int i = 0; i < piece_nums; ++i) {
        int i8 = i * 8;
        cMats[i].row(0) << trajMsg.coef_x[i8 + 0], trajMsg.coef_x[i8 + 1], trajMsg.coef_x[i8 + 2], trajMsg.coef_x[i8 + 3],
            trajMsg.coef_x[i8 + 4], trajMsg.coef_x[i8 + 5], trajMsg.coef_x[i8 + 6], trajMsg.coef_x[i8 + 7];
        cMats[i].row(1) << trajMsg.coef_y[i8 + 0], trajMsg.coef_y[i8 + 1], trajMsg.coef_y[i8 + 2], trajMsg.coef_y[i8 + 3],
            trajMsg.coef_y[i8 + 4], trajMsg.coef_y[i8 + 5], trajMsg.coef_y[i8 + 6], trajMsg.coef_y[i8 + 7];
        cMats[i].row(2) << trajMsg.coef_z[i8 + 0], trajMsg.coef_z[i8 + 1], trajMsg.coef_z[i8 + 2], trajMsg.coef_z[i8 + 3],
            trajMsg.coef_z[i8 + 4], trajMsg.coef_z[i8 + 5], trajMsg.coef_z[i8 + 6], trajMsg.coef_z[i8 + 7];

        dura[i] = trajMsg.duration[i];
      }
      for (int i = 0; i < piece_nums; ++i) {
        cMats_angle[i].row(0) << trajMsg.coef_psi[4 * i + 0], trajMsg.coef_psi[4 * i + 1], 
            trajMsg.coef_psi[4 * i + 2], trajMsg.coef_psi[4 * i + 3]; 
      }

      if (trajMsg.type == 2){
        Trajectory<7> traj(dura, cMats, cMats_angle);
        if (t > traj.getTotalDuration()) {
          // ROS_ERROR("[traj_server] trajectory too short left!");
          return false;
        }
        p = traj.getPos(t);
        v = traj.getVel(t);
        a = traj.getAcc(t);
        yaw = traj.getAngle(t);
        dyaw = traj.getAngleRate(t);
      }else{
        Trajectory<7> traj(dura, cMats);
        if (t > traj.getTotalDuration()) {
          // ROS_ERROR("[traj_server] trajectory too short left!");
          return false;
        }
        p = traj.getPos(t);
        v = traj.getVel(t);
        a = traj.getAcc(t);

        // NOTE yaw
        yaw = trajMsg.yaw;
        double d_yaw = yaw - last_yaw_;
        d_yaw = d_yaw >= M_PI ? d_yaw - 2 * M_PI : d_yaw;
        d_yaw = d_yaw <= -M_PI ? d_yaw + 2 * M_PI : d_yaw;
        double d_yaw_abs = fabs(d_yaw);
        if (d_yaw_abs >= 0.02) {
          yaw = last_yaw_ + d_yaw / d_yaw_abs * 0.02;
        }
      }

    }else if (trajMsg.order == 5){
      std::vector<Piece<5>::CoefficientMat> cMats(piece_nums);
      std::vector<AngleCoefficientMat> cMats_angle(piece_nums);
      for (int i = 0; i < piece_nums; ++i) {
        int i6 = i * 6;
        cMats[i].row(0) << trajMsg.coef_x[i6 + 0], trajMsg.coef_x[i6 + 1], trajMsg.coef_x[i6 + 2],
            trajMsg.coef_x[i6 + 3], trajMsg.coef_x[i6 + 4], trajMsg.coef_x[i6 + 5];
        cMats[i].row(1) << trajMsg.coef_y[i6 + 0], trajMsg.coef_y[i6 + 1], trajMsg.coef_y[i6 + 2],
            trajMsg.coef_y[i6 + 3], trajMsg.coef_y[i6 + 4], trajMsg.coef_y[i6 + 5];
        cMats[i].row(2) << trajMsg.coef_z[i6 + 0], trajMsg.coef_z[i6 + 1], trajMsg.coef_z[i6 + 2],
            trajMsg.coef_z[i6 + 3], trajMsg.coef_z[i6 + 4], trajMsg.coef_z[i6 + 5];

        dura[i] = trajMsg.duration[i];
      }
      for (int i = 0; i < piece_nums; ++i) {
        cMats_angle[i].row(0) << trajMsg.coef_psi[4 * i + 0], trajMsg.coef_psi[4 * i + 1], 
            trajMsg.coef_psi[4 * i + 2], trajMsg.coef_psi[4 * i + 3]; 
      }

      if (trajMsg.type == 2){
        Trajectory<5> traj(dura, cMats, cMats_angle);
        if (t > traj.getTotalDuration()) {
          // ROS_ERROR("[traj_server] trajectory too short left!");
          return false;
        }
        p = traj.getPos(t);
        v = traj.getVel(t);
        a = traj.getAcc(t);
        yaw = traj.getAngle(t);
        dyaw = traj.getAngleRate(t);
      }else{
        Trajectory<5> traj(dura, cMats);
        if (t > traj.getTotalDuration()) {
          // ROS_ERROR("[traj_server] trajectory too short left!");
          return false;
        }
        p = traj.getPos(t);
        v = traj.getVel(t);
        a = traj.getAcc(t);

        // NOTE yaw
        yaw = trajMsg.yaw;
        double d_yaw = yaw - last_yaw_;
        d_yaw = d_yaw >= M_PI ? d_yaw - 2 * M_PI : d_yaw;
        d_yaw = d_yaw <= -M_PI ? d_yaw + 2 * M_PI : d_yaw;
        double d_yaw_abs = fabs(d_yaw);
        if (d_yaw_abs >= 0.02) {
          yaw = last_yaw_ + d_yaw / d_yaw_abs * 0.02;
        }
      }



    }else{
      ROS_ERROR("[traj_server] Only support trajectory order equals 5 or 7 now!");
    }


    // if(trajMsg.traj_id > last_id_){
    //   last_id_ = trajMsg.traj_id;
    //   std::cout << "Traj_" <<  trajMsg.traj_id << std::endl;
    //   std::cout << "init: " << std::endl;
    //   std::cout << "p: " << traj.getPos(0.0).transpose() << std::endl;
    //   std::cout << "v: " << traj.getVel(0.0).transpose() << std::endl;
    //   std::cout << "a: " << traj.getAcc(0.0).transpose() << std::endl;
    // }

    publish_cmd(trajMsg.traj_id, p, v, a, yaw, dyaw);  // TODO yaw
    last_yaw_ = yaw;
    return true;
  }
  return false;
}

void heartbeatCallback(const std_msgs::EmptyConstPtr &msg) {
  heartbeat_time_ = ros::Time::now();
}

void polyTrajCallback(const quadrotor_msgs::PolyTrajConstPtr &msgPtr) {
  trajMsg_ = *msgPtr;
  if (!receive_traj_) {
    trajMsg_last_ = trajMsg_;
    receive_traj_ = true;
  }
}

void cmdCallback(const ros::TimerEvent &e) {
  if (!receive_traj_) {
    return;
  }
  ros::Time time_now = ros::Time::now();
  if ((time_now - heartbeat_time_).toSec() > 1.1) {
    ROS_ERROR_ONCE("[traj_server] Lost heartbeat from the planner, is he dead?");
    // publish_cmd(trajMsg_.traj_id, last_p_, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), 0, 0);  // TODO yaw
    // return;
  }
  if (exe_traj(trajMsg_)) {
    trajMsg_last_ = trajMsg_;
    return;
  } else if (exe_traj(trajMsg_last_)) {
    return;
  }
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "traj_server");
  ros::NodeHandle nh("~");

  ros::Subscriber poly_traj_sub = nh.subscribe("trajectory", 10, polyTrajCallback);
  ros::Subscriber heartbeat_sub = nh.subscribe("heartbeat", 10, heartbeatCallback);

  pos_cmd_pub_ = nh.advertise<quadrotor_msgs::PositionCommand>("position_cmd", 50);

  ros::Timer cmd_timer = nh.createTimer(ros::Duration(0.01), cmdCallback);

  ros::Duration(1.0).sleep();

  ROS_WARN("[Traj server]: ready.");

  ros::spin();

  return 0;
}