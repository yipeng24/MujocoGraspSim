#include <string.h>

#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/StdVector>
#include <iostream>

#include "armadillo"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/PoseWithCovarianceStamped.h"
#include "nav_msgs/Odometry.h"
#include "nav_msgs/Path.h"
#include "pose_utils.h"
#include "quadrotor_msgs/PositionCommand.h"
#include "ros/ros.h"
#include "sensor_msgs/Range.h"
#include "tf/transform_broadcaster.h"
#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"
#include "std_msgs/Float32.h"

using namespace arma;
using namespace std;

static string mesh_resource;
static double color_r, color_g, color_b, color_a, cov_scale, scale;
bool cross_config = false;
bool tf45 = false;
bool cov_pos = false;
bool cov_vel = false;
bool cov_color = false;
bool origin = false;
bool isOriginSet = false;

Eigen::Vector3d tar_odom_;
bool is_has_tar_ = false;
double pixel_occ_dis_ = 10.0;
double arm_pitch_ang_ = 0.0;
double arm_roll_ang_ = 0.0;

colvec poseOrigin(6);
ros::Publisher posePub;
ros::Publisher odomPub;
ros::Publisher pathPub;
ros::Publisher velPub;
ros::Publisher covPub;
ros::Publisher covVelPub;
ros::Publisher trajPub;
ros::Publisher sensorPub;
ros::Publisher meshPub;
ros::Publisher heightPub;
ros::Publisher fov_pub_;
ros::Publisher fov_edge_Pub;
ros::Publisher line_pub_, occ_line_pub_;
ros::Publisher clutter_hand_marker_pub_;

tf::TransformBroadcaster* broadcaster;
geometry_msgs::PoseStamped poseROS;
nav_msgs::Path pathROS;
visualization_msgs::Marker velROS;
visualization_msgs::Marker covROS;
visualization_msgs::Marker covVelROS;
visualization_msgs::Marker trajROS;
visualization_msgs::Marker sensorROS;
visualization_msgs::Marker meshROS;
sensor_msgs::Range heightROS;
string _frame_id;

// fov visualize
double max_dis_ = 3.0;
// double x_max_dis_gain_ = 0.64;
// double y_max_dis_gain_ = 0.82;
double x_max_dis_gain_ = 0.75;
double y_max_dis_gain_ = 1.0;

visualization_msgs::Marker markerNode_fov;
visualization_msgs::Marker markerEdge_fov;
visualization_msgs::Marker marker_line, fast_marker_line;
std::vector<Eigen::Vector3d> fov_node;

nav_msgs::Odometry new_odom_msg;

void fov_visual_init(std::string msg_frame_id) {
  double x_max_dis = max_dis_ * x_max_dis_gain_;
  double y_max_dis = max_dis_ * y_max_dis_gain_;

  fov_node.resize(5);
  fov_node[0][0] = 0;
  fov_node[0][1] = 0;
  fov_node[0][2] = 0;

  fov_node[1][2] = x_max_dis;
  fov_node[1][1] = y_max_dis;
  fov_node[1][0] = max_dis_;

  fov_node[2][2] = x_max_dis;
  fov_node[2][1] = -y_max_dis;
  fov_node[2][0] = max_dis_;

  fov_node[3][2] = -x_max_dis;
  fov_node[3][1] = -y_max_dis;
  fov_node[3][0] = max_dis_;

  fov_node[4][2] = -x_max_dis;
  fov_node[4][1] = y_max_dis;
  fov_node[4][0] = max_dis_;

  markerNode_fov.header.frame_id = msg_frame_id;
  // markerNode_fov.header.stamp = msg_time;
  markerNode_fov.action = visualization_msgs::Marker::ADD;
  markerNode_fov.type = visualization_msgs::Marker::SPHERE_LIST;
  markerNode_fov.ns = "fov_nodes";
  // markerNode_fov.id = 0;
  markerNode_fov.pose.orientation.w = 1;
  markerNode_fov.scale.x = 0.05;
  markerNode_fov.scale.y = 0.05;
  markerNode_fov.scale.z = 0.05;
  markerNode_fov.color.r = 0;
  markerNode_fov.color.g = 0.8;
  markerNode_fov.color.b = 1;
  markerNode_fov.color.a = 1;

  markerEdge_fov.header.frame_id = msg_frame_id;
  // markerEdge_fov.header.stamp = msg_time;
  markerEdge_fov.action = visualization_msgs::Marker::ADD;
  markerEdge_fov.type = visualization_msgs::Marker::LINE_LIST;
  markerEdge_fov.ns = "fov_edges";
  // markerEdge_fov.id = 0;
  markerEdge_fov.pose.orientation.w = 1;
  markerEdge_fov.scale.x = 0.05;
  markerEdge_fov.color.r = 0.5f;
  markerEdge_fov.color.g = 0.0;
  markerEdge_fov.color.b = 0.0;
  markerEdge_fov.color.a = 1;
}

void pub_fov_visual(Eigen::Vector3d& p, Eigen::Quaterniond& q, const ros::Time& s) {
  visualization_msgs::Marker clear_previous_msg;
  clear_previous_msg.action = visualization_msgs::Marker::DELETEALL;

  visualization_msgs::MarkerArray markerArray_fov;
  markerNode_fov.points.clear();
  markerEdge_fov.points.clear();

  std::vector<geometry_msgs::Point> fov_node_marker;
  for (int i = 0; i < (int)fov_node.size(); i++) {
    Eigen::Vector3d vector_temp;
    vector_temp = q * fov_node[i] + p;
    geometry_msgs::Point point_temp;
    point_temp.x = vector_temp[0];
    point_temp.y = vector_temp[1];
    point_temp.z = vector_temp[2];
    fov_node_marker.push_back(point_temp);
  }

  // markerNode_fov.points.push_back(fov_node_marker[0]);
  // markerNode_fov.points.push_back(fov_node_marker[1]);
  // markerNode_fov.points.push_back(fov_node_marker[2]);
  // markerNode_fov.points.push_back(fov_node_marker[3]);
  // markerNode_fov.points.push_back(fov_node_marker[4]);

  markerEdge_fov.points.push_back(fov_node_marker[0]);
  markerEdge_fov.points.push_back(fov_node_marker[1]);

  markerEdge_fov.points.push_back(fov_node_marker[0]);
  markerEdge_fov.points.push_back(fov_node_marker[2]);

  markerEdge_fov.points.push_back(fov_node_marker[0]);
  markerEdge_fov.points.push_back(fov_node_marker[3]);

  markerEdge_fov.points.push_back(fov_node_marker[0]);
  markerEdge_fov.points.push_back(fov_node_marker[4]);

  markerEdge_fov.points.push_back(fov_node_marker[0]);
  markerEdge_fov.points.push_back(fov_node_marker[1]);

  markerEdge_fov.points.push_back(fov_node_marker[1]);
  markerEdge_fov.points.push_back(fov_node_marker[2]);

  markerEdge_fov.points.push_back(fov_node_marker[2]);
  markerEdge_fov.points.push_back(fov_node_marker[3]);

  markerEdge_fov.points.push_back(fov_node_marker[3]);
  markerEdge_fov.points.push_back(fov_node_marker[4]);

  markerEdge_fov.points.push_back(fov_node_marker[4]);
  markerEdge_fov.points.push_back(fov_node_marker[1]);

  // markerArray_fov.markers.push_back(clear_previous_msg);
  markerArray_fov.markers.push_back(markerNode_fov);
  markerArray_fov.markers.push_back(markerEdge_fov);
  // fov_pub_.publish(markerArray_fov);
  markerEdge_fov.header.stamp = s;
  fov_edge_Pub.publish(clear_previous_msg);
  fov_edge_Pub.publish(markerEdge_fov);
}

  void setMarkerColor(visualization_msgs::Marker& marker,
                      double a,
                      double r,
                      double g,
                      double b) {
    marker.color.a = a;
    marker.color.r = r;
    marker.color.g = g;
    marker.color.b = b;
  }

  void setMarkerScale(visualization_msgs::Marker& marker,
                      const double& x,
                      const double& y,
                      const double& z) {
    marker.scale.x = x;
    marker.scale.y = y;
    marker.scale.z = z;
  }

  void setMarkerPose(visualization_msgs::Marker& marker,
                     const double& x,
                     const double& y,
                     const double& z) {
    marker.pose.position.x = x;
    marker.pose.position.y = y;
    marker.pose.position.z = z;
    marker.pose.orientation.w = 1;
    marker.pose.orientation.x = 0;
    marker.pose.orientation.y = 0;
    marker.pose.orientation.z = 0;
  }

void odom_tar_callback(const nav_msgs::Odometry::ConstPtr& msg) {
  if (msg->header.frame_id == string("null"))
    return;
  tar_odom_(0) = msg->pose.pose.position.x;
  tar_odom_(1) = msg->pose.pose.position.y;
  tar_odom_(2) = msg->pose.pose.position.z;

  new_odom_msg = *msg;

  is_has_tar_ = true;
}

void pixocc_callback(const std_msgs::Float32::ConstPtr& msg) {
  pixel_occ_dis_ = msg->data;
}

Eigen::Vector3d frame1toframe0(const Eigen::Vector3d& p_f1, const Eigen::Matrix3d& R_f1tof0, const Eigen::Vector3d& T_f1tof0) {
  return R_f1tof0 * p_f1 + T_f1tof0;
}

void odom_callback(const nav_msgs::Odometry::ConstPtr& msg) {

  if (msg->header.frame_id == string("null"))
    return;
  colvec pose(6);
  colvec q(4);
  colvec vel(3);

  pose(0) = msg->pose.pose.position.x;
  pose(1) = msg->pose.pose.position.y;
  pose(2) = msg->pose.pose.position.z;
  q(0) = msg->pose.pose.orientation.w;
  q(1) = msg->pose.pose.orientation.x;
  q(2) = msg->pose.pose.orientation.y;
  q(3) = msg->pose.pose.orientation.z;
  pose.rows(3, 5) = R_to_ypr(quaternion_to_R(q));
  vel(0) = msg->twist.twist.linear.x;
  vel(1) = msg->twist.twist.linear.y;
  vel(2) = msg->twist.twist.linear.z;

  // NOTE fov
  Eigen::Vector3d fov_p(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
  Eigen::Quaterniond fov_q(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
  pub_fov_visual(fov_p, fov_q, msg->header.stamp);

  if (origin && !isOriginSet) {
    isOriginSet = true;
    poseOrigin = pose;
  }
  if (origin) {
    vel = trans(ypr_to_R(pose.rows(3, 5))) * vel;
    pose = pose_update(pose_inverse(poseOrigin), pose);
    vel = ypr_to_R(pose.rows(3, 5)) * vel;
  }

  // Mesh model
  meshROS.header.frame_id = _frame_id;
  meshROS.header.stamp = msg->header.stamp;
  meshROS.ns = "mesh";
  meshROS.id = 0;
  meshROS.type = visualization_msgs::Marker::MESH_RESOURCE;
  meshROS.action = visualization_msgs::Marker::ADD;
  meshROS.mesh_use_embedded_materials = true;
  // meshROS.pose.position.x = msg->pose.pose.position.x;
  // meshROS.pose.position.y = msg->pose.pose.position.y;
  // meshROS.pose.position.z = msg->pose.pose.position.z;
  // q(0) = msg->pose.pose.orientation.w;
  // q(1) = msg->pose.pose.orientation.x;
  // q(2) = msg->pose.pose.orientation.y;
  // q(3) = msg->pose.pose.orientation.z;
  meshROS.pose.position.x = pose(0);
  meshROS.pose.position.y = pose(1);
  meshROS.pose.position.z = pose(2);

  if (cross_config) {
    colvec ypr = R_to_ypr(quaternion_to_R(q));
    ypr(0) += 45.0 * PI / 180.0;
    q = R_to_quaternion(ypr_to_R(ypr));
  }

  meshROS.pose.orientation.w = q(0);
  meshROS.pose.orientation.x = q(1);
  meshROS.pose.orientation.y = q(2);
  meshROS.pose.orientation.z = q(3);
  meshROS.scale.x = scale;
  meshROS.scale.y = scale;
  meshROS.scale.z = scale;
  // meshROS.color.a = color_a;
  // meshROS.color.r = color_r;
  // meshROS.color.g = color_g;
  // meshROS.color.b = color_b;
  meshROS.color.a = 0;
  meshROS.color.r = 0;
  meshROS.color.g = 0;
  meshROS.color.b = 0;
  meshROS.mesh_resource = mesh_resource;
  meshPub.publish(meshROS);


  //! clutter_hand_marker
//改用ch_rc_sdf 发
    visualization_msgs::MarkerArray marker_array;

    // Create the disk
    visualization_msgs::Marker disk;
    disk.header.frame_id = _frame_id;
    disk.header.stamp = msg->header.stamp;
    disk.ns = "shapes";
    disk.id = 0;
    disk.type = visualization_msgs::Marker::CYLINDER;
    disk.action = visualization_msgs::Marker::ADD;
    disk.pose.position.x = pose(0);
    disk.pose.position.y = pose(1);
    disk.pose.position.z = pose(2);
    disk.pose.orientation.w = q(0);
    disk.pose.orientation.x = q(1);
    disk.pose.orientation.y = q(2);
    disk.pose.orientation.z = q(3);
    disk.scale.x = 0.8;  // Diameter of the disk
    disk.scale.y = 0.8;  // Diameter of the disk
    disk.scale.z = 0.01;  // Height of the disk
    disk.color.r = 0.0f;
    disk.color.g = 1.0f;
    disk.color.b = 0.0f;
    disk.color.a = 0.4;
    marker_array.markers.push_back(disk);


    // Create the lines
    // visualization_msgs::Marker line;
    // line.header.frame_id = _frame_id;
    // line.header.stamp = ros::Time::now();
    // line.ns = "shapes";
    // line.type = visualization_msgs::Marker::LINE_STRIP;
    // line.action = visualization_msgs::Marker::ADD;
    // line.scale.x = 0.05;  // Line width
    // line.color.r = 1.0f;
    // line.color.g = 0.0f;
    // line.color.b = 0.0f;
    // line.color.a = 1.0;
    // line.pose.orientation.w = 1;
    // line.pose.orientation.x = 0;
    // line.pose.orientation.y = 0;
    // line.pose.orientation.z = 0;

    // std::vector<geometry_msgs::Point> points;
    // geometry_msgs::Point p;

    // double arm_l0 = 0.2;
    // double arm_l1 = 0.3;
    // double arm_l2 = 0.2;
    // arm_l2 *= 0.707;

    // Eigen::Vector3d T_f12f0, T_f02b;
    // Eigen::Matrix3d R_f12f0, R_f02b;

    // T_f12f0 << arm_l1, 0, 0;
    // T_f02b << 0, 0, -arm_l0;

    // R_f12f0 = Eigen::AngleAxisd(0,Eigen::Vector3d::UnitZ())*
    //           Eigen::AngleAxisd(0,Eigen::Vector3d::UnitY())*
    //           Eigen::AngleAxisd(arm_roll_ang_,Eigen::Vector3d::UnitX());

    // R_f02b = Eigen::AngleAxisd(0,Eigen::Vector3d::UnitZ())*
    //           Eigen::AngleAxisd(arm_pitch_ang_,Eigen::Vector3d::UnitY())*
    //           Eigen::AngleAxisd(0,Eigen::Vector3d::UnitX());

    // // p3              frame_body
    // // \               |p0
    // //   \p2           |
    // //    -------------p1
    // //   /frame1       frame0      
    // // /  
    // // p4

    // // 计算body下坐标
    // Eigen::Vector3d p4, p3, p2, p1, p0;
    // p0 << pose(0), pose(1), pose(2);
    // p1 = Eigen::Vector3d(0, 0, -arm_l0);
    // p2 = frame1toframe0(Eigen::Vector3d(arm_l1, 0, 0), R_f02b, T_f02b);

    // p3 = frame1toframe0(Eigen::Vector3d(arm_l2,-arm_l2, 0), R_f12f0, T_f12f0);
    // p3 = frame1toframe0(p3, R_f02b, T_f02b);

    // p4 = frame1toframe0(Eigen::Vector3d(arm_l2, arm_l2, 0), R_f12f0, T_f12f0);
    // p4 = frame1toframe0(p4, R_f02b, T_f02b);

    // // 1th line segment
    // line.id = 1;
    // p.x = p0.x(); p.y = p0.y(); p.z = p0.z();
    // points.push_back(p);

    // p1 = p0 + fov_q * p1;
    // p.x = p1.x(); p.y = p1.y(); p.z = p1.z();
    // points.push_back(p);
    // line.points = points;
    // marker_array.markers.push_back(line);
    // points.clear();

    // // 2th line segment
    // line.id = 2;
    // p.x = p1.x(); p.y = p1.y(); p.z = p1.z();
    // points.push_back(p);

    // p2 = p0 + fov_q * p2;
    // p.x = p2.x(); p.y = p2.y(); p.z = p2.z();
    // points.push_back(p);

    // line.points = points;
    // line.color.r = 0.0f;
    // line.color.g = 0.0f;
    // line.color.b = 1.0f;
    // marker_array.markers.push_back(line);
    // points.clear();

    // // 3th line segment
    // line.id = 3;
    // p.x = p2.x(); p.y = p2.y(); p.z = p2.z();
    // points.push_back(p);

    // p3 = p0 + fov_q * p3;
    // p.x = p3.x(); p.y = p3.y(); p.z = p3.z();
    // points.push_back(p);

    // line.points = points;
    // line.color.r = 0.0f;
    // line.color.g = 1.0f;
    // line.color.b = 0.0f;
    // marker_array.markers.push_back(line);
    // points.clear();

    // // 4th line segment
    // line.id = 4;
    // p.x = p2.x(); p.y = p2.y(); p.z = p2.z();
    // points.push_back(p);

    // p4 = p0 + fov_q * p4;
    // p.x = p4.x(); p.y = p4.y(); p.z = p4.z();
    // points.push_back(p);

    // line.points = points;
    // line.color.r = 1.0f;
    // line.color.g = 1.0f;
    // line.color.b = 0.0f;
    // marker_array.markers.push_back(line);
    // points.clear();

    // clutter_hand_marker_pub_.publish(marker_array);


  //! clutter_hand_marker


  // Pose
  poseROS.header = msg->header;
  poseROS.header.stamp = msg->header.stamp;
  poseROS.header.frame_id = string("world");
  poseROS.pose.position.x = pose(0);
  poseROS.pose.position.y = pose(1);
  poseROS.pose.position.z = pose(2);
  q = R_to_quaternion(ypr_to_R(pose.rows(3, 5)));
  poseROS.pose.orientation.w = q(0);
  poseROS.pose.orientation.x = q(1);
  poseROS.pose.orientation.y = q(2);
  poseROS.pose.orientation.z = q(3);
  posePub.publish(poseROS);

  // Velocity
  colvec yprVel(3);
  yprVel(0) = atan2(vel(1), vel(0));
  yprVel(1) = -atan2(vel(2), norm(vel.rows(0, 1), 2));
  yprVel(2) = 0;
  q = R_to_quaternion(ypr_to_R(yprVel));
  velROS.header.frame_id = string("world");
  velROS.header.stamp = msg->header.stamp;
  velROS.ns = string("velocity");
  velROS.id = 0;
  velROS.type = visualization_msgs::Marker::ARROW;
  velROS.action = visualization_msgs::Marker::ADD;
  velROS.pose.position.x = pose(0);
  velROS.pose.position.y = pose(1);
  velROS.pose.position.z = pose(2);
  velROS.pose.orientation.w = q(0);
  velROS.pose.orientation.x = q(1);
  velROS.pose.orientation.y = q(2);
  velROS.pose.orientation.z = q(3);
  velROS.scale.x = norm(vel, 2);
  velROS.scale.y = 0.05;
  velROS.scale.z = 0.05;
  velROS.color.a = 1.0;
  velROS.color.r = color_r;
  velROS.color.g = color_g;
  velROS.color.b = color_b;
  velPub.publish(velROS);

  // Path
  static ros::Time prevt = msg->header.stamp;
  if ((msg->header.stamp - prevt).toSec() > 0.1) {
    prevt = msg->header.stamp;
    pathROS.header = poseROS.header;
    pathROS.poses.push_back(poseROS);
    pathPub.publish(pathROS);
  }

  // Covariance color
  double r = 1;
  double g = 1;
  double b = 1;
  bool G = msg->twist.covariance[33];
  bool V = msg->twist.covariance[34];
  bool L = msg->twist.covariance[35];
  if (cov_color) {
    r = G;
    g = V;
    b = L;
  }

  // Covariance Position
  if (cov_pos) {
    mat P(6, 6);
    for (int j = 0; j < 6; j++)
      for (int i = 0; i < 6; i++)
        P(i, j) = msg->pose.covariance[i + j * 6];
    colvec eigVal;
    mat eigVec;
    eig_sym(eigVal, eigVec, P.submat(0, 0, 2, 2));
    if (det(eigVec) < 0) {
      for (int k = 0; k < 3; k++) {
        mat eigVecRev = eigVec;
        eigVecRev.col(k) *= -1;
        if (det(eigVecRev) > 0) {
          eigVec = eigVecRev;
          break;
        }
      }
    }
    covROS.header.frame_id = string("world");
    covROS.header.stamp = msg->header.stamp;
    covROS.ns = string("covariance");
    covROS.id = 0;
    covROS.type = visualization_msgs::Marker::SPHERE;
    covROS.action = visualization_msgs::Marker::ADD;
    covROS.pose.position.x = pose(0);
    covROS.pose.position.y = pose(1);
    covROS.pose.position.z = pose(2);
    q = R_to_quaternion(eigVec);
    covROS.pose.orientation.w = q(0);
    covROS.pose.orientation.x = q(1);
    covROS.pose.orientation.y = q(2);
    covROS.pose.orientation.z = q(3);
    covROS.scale.x = sqrt(eigVal(0)) * cov_scale;
    covROS.scale.y = sqrt(eigVal(1)) * cov_scale;
    covROS.scale.z = sqrt(eigVal(2)) * cov_scale;
    covROS.color.a = 0.4;
    covROS.color.r = r * 0.5;
    covROS.color.g = g * 0.5;
    covROS.color.b = b * 0.5;
    covPub.publish(covROS);
  }

  // Covariance Velocity
  if (cov_vel) {
    mat P(3, 3);
    for (int j = 0; j < 3; j++)
      for (int i = 0; i < 3; i++)
        P(i, j) = msg->twist.covariance[i + j * 6];
    mat R = ypr_to_R(pose.rows(3, 5));
    P = R * P * trans(R);
    colvec eigVal;
    mat eigVec;
    eig_sym(eigVal, eigVec, P);
    if (det(eigVec) < 0) {
      for (int k = 0; k < 3; k++) {
        mat eigVecRev = eigVec;
        eigVecRev.col(k) *= -1;
        if (det(eigVecRev) > 0) {
          eigVec = eigVecRev;
          break;
        }
      }
    }
    covVelROS.header.frame_id = string("world");
    covVelROS.header.stamp = msg->header.stamp;
    covVelROS.ns = string("covariance_velocity");
    covVelROS.id = 0;
    covVelROS.type = visualization_msgs::Marker::SPHERE;
    covVelROS.action = visualization_msgs::Marker::ADD;
    covVelROS.pose.position.x = pose(0);
    covVelROS.pose.position.y = pose(1);
    covVelROS.pose.position.z = pose(2);
    q = R_to_quaternion(eigVec);
    covVelROS.pose.orientation.w = q(0);
    covVelROS.pose.orientation.x = q(1);
    covVelROS.pose.orientation.y = q(2);
    covVelROS.pose.orientation.z = q(3);
    covVelROS.scale.x = sqrt(eigVal(0)) * cov_scale;
    covVelROS.scale.y = sqrt(eigVal(1)) * cov_scale;
    covVelROS.scale.z = sqrt(eigVal(2)) * cov_scale;
    covVelROS.color.a = 0.4;
    covVelROS.color.r = r;
    covVelROS.color.g = g;
    covVelROS.color.b = b;
    covVelPub.publish(covVelROS);
  }

  // Color Coded Trajectory
  static colvec ppose = pose;
  static ros::Time pt = msg->header.stamp;
  ros::Time t = msg->header.stamp;
  if ((t - pt).toSec() > 0.5) {
    trajROS.header.frame_id = string("world");
    trajROS.header.stamp = ros::Time::now();
    trajROS.ns = string("trajectory");
    trajROS.type = visualization_msgs::Marker::LINE_LIST;
    trajROS.action = visualization_msgs::Marker::ADD;
    trajROS.pose.position.x = 0;
    trajROS.pose.position.y = 0;
    trajROS.pose.position.z = 0;
    trajROS.pose.orientation.w = 1;
    trajROS.pose.orientation.x = 0;
    trajROS.pose.orientation.y = 0;
    trajROS.pose.orientation.z = 0;
    trajROS.scale.x = 0.1;
    trajROS.scale.y = 0;
    trajROS.scale.z = 0;
    trajROS.color.r = 0.0;
    trajROS.color.g = 1.0;
    trajROS.color.b = 0.0;
    trajROS.color.a = 0.8;
    geometry_msgs::Point p;
    p.x = ppose(0);
    p.y = ppose(1);
    p.z = ppose(2);
    trajROS.points.push_back(p);
    p.x = pose(0);
    p.y = pose(1);
    p.z = pose(2);
    trajROS.points.push_back(p);
    std_msgs::ColorRGBA color;
    color.r = r;
    color.g = g;
    color.b = b;
    color.a = 1;
    trajROS.colors.push_back(color);
    trajROS.colors.push_back(color);
    ppose = pose;
    pt = t;
    trajPub.publish(trajROS);
  }

  // Sensor availability
  sensorROS.header.frame_id = string("world");
  sensorROS.header.stamp = msg->header.stamp;
  sensorROS.ns = string("sensor");
  sensorROS.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  sensorROS.action = visualization_msgs::Marker::ADD;
  sensorROS.pose.position.x = pose(0);
  sensorROS.pose.position.y = pose(1);
  sensorROS.pose.position.z = pose(2);
  sensorROS.pose.orientation.w = q(0);
  sensorROS.pose.orientation.x = q(1);
  sensorROS.pose.orientation.y = q(2);
  sensorROS.pose.orientation.z = q(3);
  string strG = G ? string(" GPS ") : string("");
  string strV = V ? string(" Vision ") : string("");
  string strL = L ? string(" Laser ") : string("");
  sensorROS.text = "| " + strG + strV + strL + " |";
  sensorROS.color.a = 1.0;
  sensorROS.color.r = 1.0;
  sensorROS.color.g = 1.0;
  sensorROS.color.b = 1.0;
  sensorROS.scale.z = 0.5;
  sensorPub.publish(sensorROS);

  // Laser height measurement
  double H = msg->twist.covariance[32];
  heightROS.header.frame_id = string("height");
  heightROS.header.stamp = msg->header.stamp;
  heightROS.radiation_type = sensor_msgs::Range::ULTRASOUND;
  heightROS.field_of_view = 5.0 * M_PI / 180.0;
  heightROS.min_range = -100;
  heightROS.max_range = 100;
  heightROS.range = H;
  heightPub.publish(heightROS);


  // line
  if (is_has_tar_){
    static std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> pairline;
    static std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> occ_pairline;
    
    static int line_cnt = 0;
    if (line_cnt++ % 20 == 0){
      tar_odom_.z() = 0.8;
      if (pixel_occ_dis_ > 0 && pixel_occ_dis_ < 0.25){
        occ_pairline.emplace_back(fov_p, tar_odom_);
      }else{
        pairline.emplace_back(fov_p, tar_odom_);
      }
    }

    visualization_msgs::Marker clear_previous_msg;
    clear_previous_msg.action = visualization_msgs::Marker::DELETEALL;
    visualization_msgs::Marker arrow_msg;
    arrow_msg.type = visualization_msgs::Marker::LINE_LIST;
    arrow_msg.action = visualization_msgs::Marker::ADD;
    arrow_msg.header.frame_id = "world";
    arrow_msg.id = 0;
    arrow_msg.points.resize(2);
    setMarkerPose(arrow_msg, 0, 0, 0);
    setMarkerScale(arrow_msg, 0.08, 0.08, 0);
    // setMarkerColor(arrow_msg, 0.5, 0.92, 0.74, 0.0);
    visualization_msgs::MarkerArray arrow_list_msg;
    arrow_list_msg.markers.reserve(1 + pairline.size());
    arrow_list_msg.markers.push_back(clear_previous_msg);

    visualization_msgs::MarkerArray occ_arrow_list_msg;
    occ_arrow_list_msg.markers.reserve(1 + pairline.size());
    occ_arrow_list_msg.markers.push_back(clear_previous_msg);


    for (const auto& arrow : pairline) {
      arrow_msg.points[0].x = arrow.first[0];
      arrow_msg.points[0].y = arrow.first[1];
      arrow_msg.points[0].z = arrow.first[2];
      arrow_msg.points[1].x = arrow.second[0];
      arrow_msg.points[1].y = arrow.second[1];
      arrow_msg.points[1].z = arrow.second[2];
      arrow_msg.color.a = 0.5;
      arrow_msg.color.r = 0.92; 
      arrow_msg.color.g = 0.74; 
      arrow_msg.color.b = 0.0; 
      arrow_list_msg.markers.push_back(arrow_msg);
      arrow_msg.id += 1;
    }
    line_pub_.publish(arrow_list_msg);

    for (const auto& arrow : occ_pairline) {
      arrow_msg.points[0].x = arrow.first[0];
      arrow_msg.points[0].y = arrow.first[1];
      arrow_msg.points[0].z = arrow.first[2];
      arrow_msg.points[1].x = arrow.second[0];
      arrow_msg.points[1].y = arrow.second[1];
      arrow_msg.points[1].z = arrow.second[2];
      arrow_msg.color.a = 0.67;
      arrow_msg.color.r = 0.92; 
      arrow_msg.color.g = 0.14; 
      arrow_msg.color.b = 0.0; 
      occ_arrow_list_msg.markers.push_back(arrow_msg);
      arrow_msg.id += 1;
    }
    occ_line_pub_.publish(occ_arrow_list_msg);
  }


  // // TF for raw sensor visualization
  // if (1) {
  //   tf::Transform transform;
  //   transform.setOrigin(tf::Vector3(pose(0), pose(1), pose(2)));
  //   transform.setRotation(tf::Quaternion(q(1), q(2), q(3), q(0)));

  //   // tf::Transform transform45;
  //   // transform45.setOrigin(tf::Vector3(0, 0, 0));
  //   // colvec y45 = zeros<colvec>(3);
  //   // y45(0) = 45.0 * M_PI / 180;
  //   // colvec q45 = R_to_quaternion(ypr_to_R(y45));
  //   // transform45.setRotation(tf::Quaternion(q45(1), q45(2), q45(3), q45(0)));

  //   // tf::Transform transform90;
  //   // transform90.setOrigin(tf::Vector3(0, 0, 0));
  //   // colvec p90 = zeros<colvec>(3);
  //   // p90(1) = 90.0 * M_PI / 180;
  //   // colvec q90 = R_to_quaternion(ypr_to_R(p90));
  //   // transform90.setRotation(tf::Quaternion(q90(1), q90(2), q90(3), q90(0)));

  //   broadcaster->sendTransform(tf::StampedTransform(transform, msg->header.stamp, string("world"), string("base")));
  //   // broadcaster->sendTransform(tf::StampedTransform(transform45, msg->header.stamp, string("/base"), string("/laser")));
  //   // broadcaster->sendTransform(tf::StampedTransform(transform45, msg->header.stamp, string("/base"), string("/vision")));
  //   // broadcaster->sendTransform(tf::StampedTransform(transform90, msg->header.stamp, string("/base"), string("/height")));
  // }
}

void cmd_callback(const quadrotor_msgs::PositionCommand cmd) {
  if (cmd.header.frame_id == string("null"))
    return;

  colvec pose(6);
  pose(0) = cmd.position.x;
  pose(1) = cmd.position.y;
  pose(2) = cmd.position.z;
  colvec q(4);
  q(0) = 1.0;
  q(1) = 0.0;
  q(2) = 0.0;
  q(3) = 0.0;
  pose.rows(3, 5) = R_to_ypr(quaternion_to_R(q));

  // Mesh model
  meshROS.header.frame_id = _frame_id;
  meshROS.header.stamp = cmd.header.stamp;
  meshROS.ns = "mesh";
  meshROS.id = 0;
  meshROS.type = visualization_msgs::Marker::MESH_RESOURCE;
  meshROS.action = visualization_msgs::Marker::ADD;
  meshROS.pose.position.x = cmd.position.x;
  meshROS.pose.position.y = cmd.position.y;
  meshROS.pose.position.z = cmd.position.z;

  if (cross_config) {
    colvec ypr = R_to_ypr(quaternion_to_R(q));
    ypr(0) += 45.0 * PI / 180.0;
    q = R_to_quaternion(ypr_to_R(ypr));
  }
  meshROS.pose.orientation.w = q(0);
  meshROS.pose.orientation.x = q(1);
  meshROS.pose.orientation.y = q(2);
  meshROS.pose.orientation.z = q(3);
  meshROS.scale.x = 1.0;
  meshROS.scale.y = 1.0;
  meshROS.scale.z = 1.0;
  meshROS.color.a = color_a;
  meshROS.color.r = color_r;
  meshROS.color.g = color_g;
  meshROS.color.b = color_b;
  meshROS.mesh_resource = mesh_resource;
  meshPub.publish(meshROS);
}

// void odom_fix_callback(const ros::TimerEvent& event) {
//   static bool is_received_ = false;
//   static ros::Time last_msg_time_;
//   static ros::Time last_get_msg_time_;
//   static ros::Time last_time_;
//   static ros::Time base_time_;
//   static std::deque<Eigen::Vector3d> odom_history_vec_;
//   static int vec_len_ = 400;
//   static Eigen::Vector3d base_p;
//   if (!is_has_tar_) return;
//   if (!is_received_){
//     is_received_ = true;
//     last_msg_time_ = new_odom_msg.header.stamp;
//     odomPub.publish(new_odom_msg);
//   }else{
//     ros::Time cur_msg_time = new_odom_msg.header.stamp;
//     double dur_sec = (cur_msg_time - last_msg_time_).toSec();
//     ros::Time cur_time = ros::Time::now();
//     Eigen::Vector3d new_p(new_odom_msg.pose.pose.position.x, new_odom_msg.pose.pose.position.y, new_odom_msg.pose.pose.position.z);
//     if (new_p.x() > 2.0){
//       odom_history_vec_.emplace_back(new_odom_msg.pose.pose.position.x, new_odom_msg.pose.pose.position.y, new_odom_msg.pose.pose.position.z);
//       if ((int)odom_history_vec_.size() > vec_len_) odom_history_vec_.pop_front();
//       odomPub.publish(new_odom_msg);
//       base_p = new_p;
//       base_time_ = cur_time;
//     }else{
//       Eigen::Vector3d v = (base_p - odom_history_vec_.front()).normalized() * 1.5;
//       double fake_dur_sec = (cur_time - base_time_).toSec();
//       Eigen::Vector3d pub_p = base_p + v * fake_dur_sec;
//       nav_msgs::Odometry fake_msg = new_odom_msg;
//       fake_msg.pose.pose.position.x = pub_p.x();
//       fake_msg.pose.pose.position.y = pub_p.y();
//       fake_msg.pose.pose.position.z = pub_p.z();
//       odomPub.publish(fake_msg);
//     }
//     last_time_ = cur_time;
//   }
// }

int main(int argc, char** argv) {
  ros::init(argc, argv, "odom_visualization");
  ros::NodeHandle n("~");

  // n.param("mesh_resource", mesh_resource, std::string("package://odom_visualization/meshes/hummingbird.mesh"));
  n.param("mesh_resource", mesh_resource, std::string("package://odom_visualization/meshes/f250.dae"));

  n.param("color/r", color_r, 1.0);
  n.param("color/g", color_g, 0.0);
  n.param("color/b", color_b, 0.0);
  n.param("color/a", color_a, 1.0);
  n.param("origin", origin, false);
  n.param("robot_scale", scale, 2.0);
  n.param("frame_id", _frame_id, string("world"));

  n.param("cross_config", cross_config, false);
  n.param("tf45", tf45, false);
  n.param("covariance_scale", cov_scale, 100.0);
  n.param("covariance_position", cov_pos, false);
  n.param("covariance_velocity", cov_vel, false);
  n.param("covariance_color", cov_color, false);

  ros::Subscriber sub_odom = n.subscribe("odom", 100, odom_callback, ros::TransportHints().tcpNoDelay());
  ros::Subscriber sub_tar_odom = n.subscribe("tar_odom", 100, odom_tar_callback, ros::TransportHints().tcpNoDelay());
  ros::Subscriber sub_cmd = n.subscribe("cmd", 100, cmd_callback);
  ros::Subscriber sub_pixocc = n.subscribe("/pixel_occ_dis", 100, pixocc_callback);

  // odomPub = n.advertise<nav_msgs::Odometry>("/target/odom", 1, false);
  // ros::Timer odom_fix_timer_;
  // odom_fix_timer_ = n.createTimer(ros::Duration(1.0 / 400.0), &odom_fix_callback);

  posePub = n.advertise<geometry_msgs::PoseStamped>("pose", 1, false);
  pathPub = n.advertise<nav_msgs::Path>("path", 100, true);
  velPub = n.advertise<visualization_msgs::Marker>("velocity", 100, true);
  covPub = n.advertise<visualization_msgs::Marker>("covariance", 100, true);
  covVelPub = n.advertise<visualization_msgs::Marker>("covariance_velocity", 100, true);
  trajPub = n.advertise<visualization_msgs::Marker>("trajectory", 100, true);
  sensorPub = n.advertise<visualization_msgs::Marker>("sensor", 100, true);
  meshPub = n.advertise<visualization_msgs::Marker>("robot", 100, true);
  heightPub = n.advertise<sensor_msgs::Range>("height", 100, true);
  fov_pub_ = n.advertise<visualization_msgs::MarkerArray>("fov_visual", 1, false);
  fov_edge_Pub = n.advertise<visualization_msgs::Marker>("fov_E", 1, true);
  clutter_hand_marker_pub_ = n.advertise<visualization_msgs::MarkerArray>("clutter_hand", true);

  line_pub_ = n.advertise<visualization_msgs::MarkerArray>("/track_line", 1, false);
  occ_line_pub_ = n.advertise<visualization_msgs::MarkerArray>("/occ_line", 1, false);

  std::cout << "odom_visualization node started" << std::endl;

  tf::TransformBroadcaster b;
  broadcaster = &b;
  fov_visual_init("world");

  ros::spin();

  return 0;
}
