#ifndef _MAP_ROS_H
#define _MAP_ROS_H

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/time_synchronizer.h>
#include <pcl_conversions/pcl_conversions.h>
#include "quadrotor_msgs/frame.h"
#include "std_msgs/Bool.h"
#include "std_msgs/Float32.h"
#include <ros/ros.h>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <nav_msgs/Odometry.h>

#include <memory>
#include <random>

using std::shared_ptr;
using std::normal_distribution;
using std::default_random_engine;

using namespace std;

namespace airgrasp {
class SDFMap;

class MapROS {
  public:
    MapROS();
    ~MapROS();
    void setMap(SDFMap* map);
    void setMapName(const std::string& map_name);
    void init();
    void registROS();
  private:
    void depthPoseCallback(const sensor_msgs::ImageConstPtr& img,
                          const nav_msgs::OdometryConstPtr& pose);
    void cloudPoseCallback(const sensor_msgs::PointCloud2ConstPtr& msg,
                          const geometry_msgs::PoseStampedConstPtr& pose);
    void updateESDFCallback(const ros::TimerEvent& /*event*/);
    void visCallback(const ros::TimerEvent& /*event*/);

    void move_base_callback(const geometry_msgs::PoseStamped::ConstPtr &msg);

    void publishMapOcc();
    void publishMapInf();
    void publishUnknown();
    void publishInfo();
    void publishLoss();
    // void publishMapUnvalid();
    void publishESDF();
    void publishDepth();

    void proessDepthImage();

    void setVisPtr(std::shared_ptr<visualization_airgrasp::Visualization> vis_ptr) {
      vis_ptr_ = vis_ptr;

      std::vector<std::pair<Eigen::Vector3d, double>> pcl_i;
      vis_ptr_->visualize_pointcloud_itensity(pcl_i, "sdf_map/inf_occ");
      vis_ptr_->visualize_pointcloud_itensity(pcl_i, "sdf_map/unknown");
      vis_ptr_->visualize_pointcloud_itensity(pcl_i, "sdf_map/unknown3d");
      vis_ptr_->visualize_pointcloud_itensity(pcl_i, "sdf_map/occ");
    }

    CameraParam getCamParam() {
      return cam_param_;
    }

  private:
    SDFMap* map_;
    // may use ExactTime?
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, nav_msgs::Odometry>
        SyncPolicyImagePose;
    typedef shared_ptr<message_filters::Synchronizer<SyncPolicyImagePose>> SynchronizerImagePose;
    // typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2,
    //                                                         geometry_msgs::PoseStamped>
    //     SyncPolicyCloudPose;
    // typedef shared_ptr<message_filters::Synchronizer<SyncPolicyCloudPose>> SynchronizerCloudPose;

    ros::NodeHandle node_;
    shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> depth_sub_;
    // shared_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> cloud_sub_;
    shared_ptr<message_filters::Subscriber<nav_msgs::Odometry>> pose_sub_;
    SynchronizerImagePose sync_image_pose_;
    // SynchronizerCloudPose sync_cloud_pose_;

    ros::Publisher map_local_pub_, map_local_inflate_pub_, esdf_pub_, map_all_pub_, unknown_pub_,
        update_range_pub_, depth_pub_, grid_CR_img_pub_,benchmark_time_pub_;
    ros::Timer esdf_timer_, vis_timer_;

    ros::Subscriber render_frame_sub_, gs_list_sub_, move_base_sub_, benchmark_start_flag_sub_;

    ros::ServiceClient cover_rate_client_, cover_rate_pcl_client_, gs_list_client_;

    
    // params, depth projection
    int depth_filter_margin_;
    double k_depth_scaling_factor_;
    int skip_pixel_;
    string frame_id_;
    // msg publication
    double esdf_slice_height_;
    double visualization_truncate_height_, visualization_truncate_low_;
    bool show_esdf_time_, show_occ_time_;
    bool show_all_map_;

    // rviz or gazebo
    bool input_cam_pose_ = true;

    // data
    // flags of map state
    bool local_updated_, esdf_need_update_, have_first_depth_, benchmark_mode_ = false;
    // input
    Eigen::Vector3d camera_pos_;
    Eigen::Quaterniond camera_q_;
    unique_ptr<cv::Mat> depth_image_;
    vector<Eigen::Vector3d> proj_points_;
    int proj_points_cnt;
    double fuse_time_, esdf_time_, max_fuse_time_, max_esdf_time_;
    int fuse_num_, esdf_num_;
    pcl::PointCloud<pcl::PointXYZ> point_cloud_;

    normal_distribution<double> rand_noise_;
    default_random_engine eng_;

    ros::Time map_start_time_;
    std::shared_ptr<visualization_airgrasp::Visualization> vis_ptr_;
    CameraParam cam_param_;

    Eigen::Matrix4d cam2inputPose_;

    std::string map_name_;

    friend SDFMap;
};
}

#endif