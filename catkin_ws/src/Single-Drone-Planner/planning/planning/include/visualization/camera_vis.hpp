#pragma once
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>

#include <Eigen/Geometry>
#include <iostream>
#include <unordered_map>
#include <string>

namespace visualization {

class CameraVis{
 private:
  ros::NodeHandle nh_;
  ros::Publisher image_pub_;

  // camera parameters
  bool is_set_ = false;
  Eigen::Matrix3d cam2body_R_;
  Eigen::Vector3d cam2body_p_;
  double fx_, fy_, cx_, cy_;

 public:
  CameraVis(ros::NodeHandle& nh, const std::string& name) : nh_(nh)
  {
    image_pub_ = nh_.advertise<sensor_msgs::Image>("/camera_img_" + name, 100);
  }

  // set the virtual camera's parameters
  void set_para(const Eigen::Matrix3d& cam2body_R,
                const Eigen::Vector3d& cam2body_p,
                const double& fx,
                const double& fy,
                const double& cx,
                const double& cy)
  {
    cam2body_R_ = cam2body_R;
    cam2body_p_ = cam2body_p;
    fx_ = fx;
    fy_ = fy;
    cx_ = cx;
    cy_ = cy;
    is_set_ = true;
  }

  bool is_tag_visible(const Eigen::Vector3d odom_p,
                      const Eigen::Quaterniond odom_q,
                      const Eigen::Vector3d tar_p)
  {
    if (!is_set_){
      ROS_ERROR_STREAM("[CameraVis] Not set the virtual camera's parameters");
      return false;
    }

    static double eps = 1e-6;
    Eigen::Quaterniond qbw;
    qbw.w() = odom_q.w(); qbw.x() = -odom_q.x(); qbw.y() = -odom_q.y(); qbw.z() = -odom_q.z();
    Eigen::Vector3d pc; // target in camera frame
    pc = cam2body_R_ * (qbw * (tar_p - odom_p) - cam2body_p_); 
    // std::cout << "pb: " << (qbw * (tar_p - odom_p)).transpose()<<", pc: "<< pc.transpose()<<std::endl;
    double u = pc.x() / (pc.z() + eps) * fx_ + cx_;
    double v = pc.y() / (pc.z() + eps) * fy_ + cy_;

    double height, width;
    height = 480;
    width = 640;
    if (u >= 0 && u < width && v >= 0 && v < height){
      return true;
    }else{
      return false;
    }
  }

  // given the ego odom, taget odom, visulize the virtual image
  bool visualize_camera_img(const Eigen::Vector3d odom_p,
                            const Eigen::Quaterniond odom_q,
                            const Eigen::Vector3d tar_p)
  {
    if (!is_set_){
      ROS_ERROR_STREAM("[CameraVis] Not set the virtual camera's parameters");
      return false;
    }

    static double eps = 1e-6;
    Eigen::Quaterniond qbw;
    qbw.w() = odom_q.w(); qbw.x() = -odom_q.x(); qbw.y() = -odom_q.y(); qbw.z() = -odom_q.z();
    Eigen::Vector3d pc; // target in camera frame
    pc = cam2body_R_.transpose() * (qbw * (tar_p - odom_p) - cam2body_p_); 
    std::cout << "pb: " << (qbw * (tar_p - odom_p)).transpose()<<", pc: "<< pc.transpose()<<std::endl;
    double u = pc.x() / (pc.z() + eps) * fx_ + cx_;
    double v = pc.y() / (pc.z() + eps) * fy_ + cy_;
    std::cout << "u: "<<u<<", v:" <<v<<std::endl;

    cv::Mat depth_mat;
    double height, width;
    height = 480;
    width = 640;

    if (!(u >= 0 && u < width && v >= 0 && v < height)){
      return false;
    }

    depth_mat = cv::Mat::zeros(height, width, CV_32FC1);
    double r = 0.08 / (pc.z() + eps) * fx_;

    // depth_mat.at<float>(floor(u), floor(v)) = 1;

    for (int i = u - r; i < u + r; i++){
      for (int j = v - r; j < v + r; j++){
        if (i >= 0 && i < width && j >= 0 && j < height && (i-u)*(i-u) + (j-v)*(j-v) <= r*r){
          // depth_mat.at<float>(i, j) = 1;
          depth_mat.at<float>(j, i) = 1;
        }
      }
    }

    cv_bridge::CvImage out_msg;
    out_msg.header.stamp = ros::Time::now();
    out_msg.header.frame_id = "camera";
    out_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    out_msg.image = depth_mat.clone();
    image_pub_.publish(out_msg.toImageMsg());

    // ROS_INFO("ImageMsg Send.");
    if (u >= 0 && u < width && v >= 0 && v < height){
      return true;
    }else{
      return false;
    }
  }

};


} // namespace visualization 