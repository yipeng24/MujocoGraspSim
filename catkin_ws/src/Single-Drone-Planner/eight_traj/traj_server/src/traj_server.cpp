#include <eigen3/Eigen/Dense>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <quadrotor_msgs/PolynomialTrajectory.h>
#include <quadrotor_msgs/PositionCommand.h>
#include <ros/console.h>
#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/tf.h>
#include <tf/transform_datatypes.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

// #define TRAJ_WITH_YAW

const int _DIM_x = 0;
const int _DIM_y = 1;
const int _DIM_z = 2;

using namespace std;

int _poly_order_min, _poly_order_max;

class TrajectoryServer {
private:
  // Subscribers
  ros::Subscriber _odom_sub;
  ros::Subscriber _traj_sub;

  // airgrasp
  ros::Subscriber _js_est_sub;
  Eigen::VectorXd _js_est_pos, _js_est_pos_start;
  Eigen::VectorXd _js_sweep_phi_start, _js_sweep_omg, _js_sweep_freq;
  Eigen::VectorXd _js_sweep_low, _js_sweep_high;
  bool _first_js_est = false;
  bool _en_js_sweep = false;

  // publishers
  ros::Publisher _cmd_pub;
  ros::Publisher _vis_cmd_pub;
  ros::Publisher _vis_vel_pub;
  ros::Publisher _vis_acc_pub;
  ros::Publisher _vis_traj_pub;

  // configuration for trajectory
  int _n_segment = 0;
  int _traj_id = 0;
  uint32_t _traj_flag = 0;
  Eigen::VectorXd _time;
  Eigen::MatrixXd _coef[3];
  vector<int> _order;

  double _vis_traj_width = 0.2;
  double mag_coeff;
  ros::Time _final_time = ros::TIME_MIN;
  ros::Time _start_time = ros::TIME_MAX;
  double _start_yaw = 0.0, _final_yaw = 0.0;

  geometry_msgs::Point hover_position;

  // state of the server
  // enum ServerState{INIT, TRAJ, HOVER} state = INIT;
  enum ServerState { INIT = 0, TRAJ, HOVER } state = INIT;
  ;
  nav_msgs::Odometry _odom;
  quadrotor_msgs::PositionCommand _cmd;
  geometry_msgs::PoseStamped _vis_cmd;

  visualization_msgs::Marker _vis_vel, _vis_acc;
  visualization_msgs::Marker _vis_traj;

public:
  vector<Eigen::VectorXd>
      CList; // Position coefficients vector, used to record all the pre-compute
             // 'n choose k' combinatorial for the bernstein coefficients .
  vector<Eigen::VectorXd> CvList; // Velocity coefficients vector.
  vector<Eigen::VectorXd> CaList; // Acceleration coefficients vector.

  TrajectoryServer(ros::NodeHandle &handle) {
    handle.param("en_js_sweep", _en_js_sweep, false);
    std::cout << "en_js_sweep:" << _en_js_sweep << std::endl;

    _js_sweep_freq.resize(3);
    handle.param("sweep_freq_0", _js_sweep_freq[0], 0.5);
    handle.param("sweep_freq_1", _js_sweep_freq[1], 0.5);
    handle.param("sweep_freq_2", _js_sweep_freq[2], 0.5);

    _js_est_sub = handle.subscribe("joint_state_est", 50,
                                   &TrajectoryServer::rcvEstJointState, this,
                                   ros::TransportHints().tcpNoDelay());

    _odom_sub =
        handle.subscribe("odometry", 50, &TrajectoryServer::rcvOdometryCallback,
                         this, ros::TransportHints().tcpNoDelay());

    _traj_sub = handle.subscribe(
        "trajectory", 2, &TrajectoryServer::rcvTrajectoryCallabck, this);

    _cmd_pub = handle.advertise<quadrotor_msgs::PositionCommand>(
        "position_command", 50);

    _vis_cmd_pub =
        handle.advertise<geometry_msgs::PoseStamped>("desired_position", 50);

    _vis_vel_pub =
        handle.advertise<visualization_msgs::Marker>("desired_velocity", 50);

    _vis_acc_pub = handle.advertise<visualization_msgs::Marker>(
        "desired_acceleration", 50);

    _vis_traj_pub =
        handle.advertise<visualization_msgs::Marker>("trajectory_vis", 1);

    double pos_gain[3] = {5.7, 5.7, 6.2};
    double vel_gain[3] = {3.4, 3.4, 4.0};
    setGains(pos_gain, vel_gain);

    // airgrasp
    if (_en_js_sweep) {
      _js_sweep_low.resize(3);
      _js_sweep_high.resize(3);
      _js_sweep_low << 0, 0, -M_PI_2;
      _js_sweep_high << M_PI_2, M_PI_2, M_PI_2;
      _js_sweep_omg = 2.0 * M_PI * _js_sweep_freq;
    }

    _vis_traj.header.stamp = ros::Time::now();
    _vis_traj.header.frame_id = "map";

    _vis_traj.ns = "trajectory/trajectory";
    _vis_traj.id = 0;
    _vis_traj.type = visualization_msgs::Marker::SPHERE_LIST;
    _vis_traj.action = visualization_msgs::Marker::ADD;
    _vis_traj.scale.x = _vis_traj_width;
    _vis_traj.scale.y = _vis_traj_width;
    _vis_traj.scale.z = _vis_traj_width;
    _vis_traj.pose.orientation.x = 0.0;
    _vis_traj.pose.orientation.y = 0.0;
    _vis_traj.pose.orientation.z = 0.0;
    _vis_traj.pose.orientation.w = 1.0;
    _vis_traj.color.r = 0.0;
    _vis_traj.color.g = 0.0;
    _vis_traj.color.b = 1.0;
    _vis_traj.color.a = 0.3;
    _vis_traj.points.clear();
  }

  void setGains(double pos_gain[3], double vel_gain[3]) {
    _cmd.kx[_DIM_x] = pos_gain[_DIM_x];
    _cmd.kx[_DIM_y] = pos_gain[_DIM_y];
    _cmd.kx[_DIM_z] = pos_gain[_DIM_z];

    _cmd.kv[_DIM_x] = vel_gain[_DIM_x];
    _cmd.kv[_DIM_y] = vel_gain[_DIM_y];
    _cmd.kv[_DIM_z] = vel_gain[_DIM_z];
  }

  bool cmd_flag = false;
  void rcvOdometryCallback(const nav_msgs::Odometry &odom) {
    // ROS_WARN("state = %d",state);

    if (odom.child_frame_id == "X" || odom.child_frame_id == "O")
      return;
    // #1. store the odometry
    _odom = odom;
    _vis_cmd.header = _odom.header;
    _vis_cmd.header.frame_id = "world";

    if (state == INIT) {
      // ROS_WARN("[TRAJ SERVER] Pub initial pos command");
      _cmd.position = _odom.pose.pose.position;

      _cmd.header.stamp = _odom.header.stamp;
      _cmd.header.frame_id = "world";
      _cmd.trajectory_flag = _traj_flag;

      _cmd.velocity.x = 0.0;
      _cmd.velocity.y = 0.0;
      _cmd.velocity.z = 0.0;

      _cmd.acceleration.x = 0.0;
      _cmd.acceleration.y = 0.0;
      _cmd.acceleration.z = 0.0;
      // _cmd_pub.publish(_cmd);

      _vis_cmd.pose.position.x = _cmd.position.x;
      _vis_cmd.pose.position.y = _cmd.position.y;
      _vis_cmd.pose.position.z = _cmd.position.z;
      // _vis_cmd_pub.publish(_vis_cmd);

      return;
    }

    // change the order between #2 and #3. zxzxzxzx

    // #2. try to calculate the new state
    if (state == TRAJ &&
        ((ros::Time::now() /*odom.header.stamp*/ - _start_time).toSec() /
             mag_coeff >
         (_final_time - _start_time).toSec())) {
      state = HOVER;
      hover_position = _odom.pose.pose.position;
      _traj_flag = quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_COMPLETED;
    }

    // #3. try to publish command
    pubPositionCommand();
  }

  void rcvEstJointState(const sensor_msgs::JointState &msg) {
    if (!_first_js_est) {
      if (msg.position.size() >= 3) {
        _js_est_pos.resize(3);
        _first_js_est = true;
        std::cout << "js est init" << std::endl;
      } else {
        std::cout << "[traj_server] The joint state is not enough."
                  << std::endl;
        return;
      }
    }

    for (size_t i = 0; i < 3; i++)
      _js_est_pos(i) = msg.position[i];
  }

  void rcvTrajectoryCallabck(const quadrotor_msgs::PolynomialTrajectory &traj) {
    // ROS_WARN("[SERVER] Recevied The Trajectory with %.3lf.",
    // _start_time.toSec()); ROS_WARN("[SERVER] Now the odom time is : ");
    //  #1. try to execuse the action

    // 防止打桨
    // if(_js_est_pos(0) < 0.1 * M_PI) {
    //   ROS_WARN("js_est_pos(0) too small, not execute traj");
    //   return;
    // }

    if (traj.action == quadrotor_msgs::PolynomialTrajectory::ACTION_ADD) {
      ROS_WARN("[traj_server] Loading the trajectory.");
      if ((int)traj.trajectory_id < 1) {
        ROS_ERROR(
            "[traj_server] The trajectory_id must start from 1"); //. zxzxzxzx
        return;
      }
      if ((int)traj.trajectory_id > 1 && (int)traj.trajectory_id < _traj_id)
        return;

      state = TRAJ;
      _traj_flag = quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_READY;
      _traj_id = traj.trajectory_id;
      _n_segment = traj.num_segment;
      _final_time = _start_time = ros::Time::now(); //_odom.header.stamp;
      _time.resize(_n_segment);

      // airgrasp
      if (_en_js_sweep && _first_js_est) {
        std::cout << "before get js_start" << std::endl;
        _js_est_pos_start = _js_est_pos;
        _js_sweep_phi_start.resize(3);
        
        // Normalize the difference to the sweep range [-1, 1]
        Eigen::Vector3d sweep_range = 0.5 * (_js_sweep_high - _js_sweep_low);
        Eigen::Vector3d normalized_diff = (_js_est_pos_start - 0.5*(_js_sweep_low + _js_sweep_high)).array() / sweep_range.array();
        
        // Clamp to [-1, 1] to prevent NaN from asin due to sensor noise
        normalized_diff = normalized_diff.cwiseMax(-1.0).cwiseMin(1.0);
        
        _js_sweep_phi_start << asin(normalized_diff(0)),
                               M_PI - asin(normalized_diff(1)), 
                               asin(normalized_diff(2));
        std::cout << "_js_sweep_phi_start:" << _js_sweep_phi_start.transpose() << std::endl;
      }

      _order.clear();
      for (int idx = 0; idx < _n_segment; ++idx) {
        _final_time += ros::Duration(traj.time[idx]);
        _time(idx) = traj.time[idx];
        _order.push_back(traj.order[idx]);
      }

      _start_yaw = traj.start_yaw;
      _final_yaw = traj.final_yaw;
      mag_coeff = traj.mag_coeff;

      int max_order = *max_element(begin(_order), end(_order));

      _coef[_DIM_x] = Eigen::MatrixXd::Zero(max_order + 1, _n_segment);
      _coef[_DIM_y] = Eigen::MatrixXd::Zero(max_order + 1, _n_segment);
      _coef[_DIM_z] = Eigen::MatrixXd::Zero(max_order + 1, _n_segment);

      // ROS_WARN("stack the coefficients");
      int shift = 0;
      for (int idx = 0; idx < _n_segment; ++idx) {
        int order = traj.order[idx];

        for (int j = 0; j < (order + 1); ++j) {
          _coef[_DIM_x](j, idx) = traj.coef_x[shift + j];
          _coef[_DIM_y](j, idx) = traj.coef_y[shift + j];
          _coef[_DIM_z](j, idx) = traj.coef_z[shift + j];
        }

        shift += (order + 1);
      }
    } else if (traj.action ==
               quadrotor_msgs::PolynomialTrajectory::ACTION_ABORT) {
      ROS_WARN("[SERVER] Aborting the trajectory.");
      state = HOVER;
      hover_position = _odom.pose.pose.position;
      _traj_flag = quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_COMPLETED;
    } else if (traj.action ==
               quadrotor_msgs::PolynomialTrajectory::ACTION_WARN_IMPOSSIBLE) {
      state = HOVER;
      hover_position = _odom.pose.pose.position;
      _traj_flag =
          quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_IMPOSSIBLE;
    }
  }

  void pubPositionCommand() {
    // #1. check if it is right state
    if (state == INIT)
      return;
    if (state == HOVER)
      return;

    _cmd.yaw_dot = 0.0;

    // #2. locate the trajectory segment
    if (state == TRAJ) {
      _cmd.header.stamp = ros::Time::now(); //_odom.header.stamp;

      _cmd.header.frame_id = "world";
      _cmd.trajectory_flag = _traj_flag;
      _cmd.trajectory_id = _traj_id;

      double t =
          max(0.0, (_cmd.header.stamp - _start_time).toSec()); // / mag_coeff;

      // cout<<"t: "<<t<<endl;
      _cmd.yaw_dot = 0.0;
      _cmd.yaw = _start_yaw + (_final_yaw - _start_yaw) * t /
                                  ((_final_time - _start_time).toSec() + 1e-9);

      // airgrasp
      if (_en_js_sweep) {
        Eigen::VectorXd js_cmd_p, js_cmd_v;

        js_cmd_p = 0.5 * (_js_sweep_low + _js_sweep_high) +
                   (0.5 * (_js_sweep_high - _js_sweep_low).array() *
                    (_js_sweep_omg * t + _js_sweep_phi_start).array().sin())
                       .matrix();
        js_cmd_v = (0.5 * (_js_sweep_high - _js_sweep_low).array() *
                    _js_sweep_omg.array() *
                    (_js_sweep_omg * t + _js_sweep_phi_start).array().cos())
                       .matrix();

        _cmd.theta.position.clear();
        _cmd.theta.velocity.clear();
        _cmd.theta.position.push_back(js_cmd_p(0));
        _cmd.theta.position.push_back(js_cmd_p(1));
        _cmd.theta.position.push_back(js_cmd_p(2));
        _cmd.theta.velocity.push_back(js_cmd_p(0));
        _cmd.theta.velocity.push_back(js_cmd_p(1));
        _cmd.theta.velocity.push_back(js_cmd_p(2));
      }

      // #3. calculate the desired states
      // ROS_WARN("[SERVER] the time : %.3lf\n, n = %d, m = %d", t, _n_order,
      // _n_segment);
      for (int idx = 0; idx < _n_segment; ++idx) {
        if (t > _time[idx] && idx + 1 < _n_segment) {
          t -= _time[idx];
        } else {
          t /= _time[idx];

          _cmd.position.x = 0.0;
          _cmd.position.y = 0.0;
          _cmd.position.z = 0.0;
          _cmd.velocity.x = 0.0;
          _cmd.velocity.y = 0.0;
          _cmd.velocity.z = 0.0;
          _cmd.acceleration.x = 0.0;
          _cmd.acceleration.y = 0.0;
          _cmd.acceleration.z = 0.0;

          int cur_order = _order[idx];
          int cur_poly_num = cur_order + 1;

          for (int i = 0; i < cur_poly_num; i++) {
            _cmd.position.x += _coef[_DIM_x].col(idx)(i) * pow(t, i);
            _cmd.position.y += _coef[_DIM_y].col(idx)(i) * pow(t, i);
            _cmd.position.z += _coef[_DIM_z].col(idx)(i) * pow(t, i);

            if (i < (cur_poly_num - 1)) {
              _cmd.velocity.x += (i + 1) * _coef[_DIM_x].col(idx)(i + 1) *
                                 pow(t, i) / _time[idx];

              _cmd.velocity.y += (i + 1) * _coef[_DIM_y].col(idx)(i + 1) *
                                 pow(t, i) / _time[idx];

              _cmd.velocity.z += (i + 1) * _coef[_DIM_z].col(idx)(i + 1) *
                                 pow(t, i) / _time[idx];
            }

            if (i < (cur_poly_num - 2)) {
              _cmd.acceleration.x += (i + 2) * (i + 1) *
                                     _coef[_DIM_x].col(idx)(i + 2) * pow(t, i) /
                                     _time[idx] / _time[idx];

              _cmd.acceleration.y += (i + 2) * (i + 1) *
                                     _coef[_DIM_y].col(idx)(i + 2) * pow(t, i) /
                                     _time[idx] / _time[idx];

              _cmd.acceleration.z += (i + 2) * (i + 1) *
                                     _coef[_DIM_z].col(idx)(i + 2) * pow(t, i) /
                                     _time[idx] / _time[idx];
            }
          }

#ifdef TRAJ_WITH_YAW
          _cmd.yaw = atan2(_cmd.velocity.y, _cmd.velocity.x);

          tf::Quaternion quat(
              _odom.pose.pose.orientation.x, _odom.pose.pose.orientation.y,
              _odom.pose.pose.orientation.z, _odom.pose.pose.orientation.w);
          tf::Matrix3x3 rotM(quat);
          double roll, pitch, yaw;
          rotM.getRPY(roll, pitch, yaw);
          const double pi = 3.14159265358979;
          double deltaYaw = yaw - _cmd.yaw;
          if (deltaYaw < -pi)
            deltaYaw += 2 * pi;
          if (deltaYaw >= pi)
            deltaYaw -= 2 * pi;

          _cmd.yaw_dot = -0.75 * deltaYaw;
          if (fabs(_cmd.yaw_dot) > pi / 2) {
            _cmd.yaw_dot /= fabs(_cmd.yaw_dot);
            _cmd.yaw_dot *= pi / 2;
          }

#else
          _cmd.yaw = 0.0;
          _cmd.yaw_dot = 0.0;
#endif
          // ROS_WARN("%.8f %.8f %.8f
          // %.8f",_cmd.velocity.x,_cmd.velocity.y,_cmd.yaw,t);

          break;
        }
      }
    }
    _cmd.grab_cmd = 2;

    // #4. just publish
    _cmd_pub.publish(_cmd);

    _vis_cmd.header = _cmd.header;
    _vis_cmd.pose.position.x = _cmd.position.x;
    _vis_cmd.pose.position.y = _cmd.position.y;
    _vis_cmd.pose.position.z = _cmd.position.z;

    tf::Quaternion q_ = tf::createQuaternionFromYaw(_cmd.yaw);
    geometry_msgs::Quaternion odom_quat;
    tf::quaternionTFToMsg(q_, odom_quat);
    _vis_cmd.pose.orientation = odom_quat;
    _vis_cmd_pub.publish(_vis_cmd);

    _vis_vel.ns = "vel";
    _vis_vel.id = 0;
    _vis_vel.header.frame_id = "world";
    _vis_vel.type = visualization_msgs::Marker::ARROW;
    _vis_vel.action = visualization_msgs::Marker::ADD;
    _vis_vel.color.a = 1.0;
    _vis_vel.color.r = 0.0;
    _vis_vel.color.g = 1.0;
    _vis_vel.color.b = 0.0;

    _vis_vel.header.stamp = _odom.header.stamp;
    _vis_vel.points.clear();

    geometry_msgs::Point pt;
    pt.x = _cmd.position.x;
    pt.y = _cmd.position.y;
    pt.z = _cmd.position.z;

    _vis_traj.points.push_back(pt);
    _vis_traj_pub.publish(_vis_traj);

    _vis_vel.points.push_back(pt);

    pt.x = _cmd.position.x + _cmd.velocity.x;
    pt.y = _cmd.position.y + _cmd.velocity.y;
    pt.z = _cmd.position.z + _cmd.velocity.z;

    _vis_vel.points.push_back(pt);

    _vis_vel.scale.x = 0.2;
    _vis_vel.scale.y = 0.4;
    _vis_vel.scale.z = 0.4;

    _vis_vel_pub.publish(_vis_vel);

    _vis_acc.ns = "acc";
    _vis_acc.id = 0;
    _vis_acc.header.frame_id = "world";
    _vis_acc.type = visualization_msgs::Marker::ARROW;
    _vis_acc.action = visualization_msgs::Marker::ADD;
    _vis_acc.color.a = 1.0;
    _vis_acc.color.r = 1.0;
    _vis_acc.color.g = 1.0;
    _vis_acc.color.b = 0.0;

    _vis_acc.header.stamp = _odom.header.stamp;

    _vis_acc.points.clear();
    pt.x = _cmd.position.x;
    pt.y = _cmd.position.y;
    pt.z = _cmd.position.z;

    _vis_acc.points.push_back(pt);

    pt.x = _cmd.position.x + _cmd.acceleration.x;
    pt.y = _cmd.position.y + _cmd.acceleration.y;
    pt.z = _cmd.position.z + _cmd.acceleration.z;

    _vis_acc.points.push_back(pt);

    _vis_acc.scale.x = 0.2;
    _vis_acc.scale.y = 0.4;
    _vis_acc.scale.z = 0.4;

    _vis_acc_pub.publish(_vis_acc);
  }
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "gradient_trajectory_server_node");
  ros::NodeHandle handle("~");

  TrajectoryServer server(handle);

  ros::spin();

  return 0;
}
