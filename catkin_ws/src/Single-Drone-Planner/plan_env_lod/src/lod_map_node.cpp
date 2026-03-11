#include <ros/ros.h>
#include <Eigen/Dense>
#include <sensor_msgs/PointCloud2.h>
#include <quadrotor_msgs/SyncFrame.h>
#include "plan_env_lod/sdf_map.h"

std::shared_ptr<airgrasp::SDFMap> tgt_map_, env_map_;

void syncFrameCallback(const quadrotor_msgs::SyncFrame::ConstPtr& msg)
{
    // 取出 body_odom
    const nav_msgs::Odometry& body_odom = msg->body_odom;
    ROS_INFO_STREAM("Body Odom position: "
                    << body_odom.pose.pose.position.x << ", "
                    << body_odom.pose.pose.position.y << ", "
                    << body_odom.pose.pose.position.z);

    Eigen::Vector3d position;
    position << msg->body_odom.pose.pose.position.x,
                msg->body_odom.pose.pose.position.y,
                msg->body_odom.pose.pose.position.z;

    std::cout << "position: " << position.transpose() << std::endl;

    pcl::PointCloud<pcl::PointXYZ> cloud_tgt, cloud_env;
    pcl::fromROSMsg(msg->tgt_pcl, cloud_tgt);
    pcl::fromROSMsg(msg->env_pcl, cloud_env);

    pcl::PointCloud<pcl::PointXYZ> cloud_combined;
    cloud_combined = cloud_tgt;
    cloud_combined += cloud_env;   // 拼接

    // tgt_map_->inputPointCloud(cloud_tgt, cloud_tgt.points.size(), position);
    // tgt_map_->inputPointCloud(cloud_env, cloud_env.points.size(), position, false);
    tgt_map_->inputPointCloud(cloud_combined, cloud_combined.points.size(), position);
    tgt_map_->vis();
    ROS_INFO_STREAM("tgt cloud num: " << cloud_tgt.points.size());

    env_map_->inputPointCloud(cloud_env, cloud_env.points.size(), position);
    env_map_->inputPointCloud(cloud_tgt, cloud_tgt.points.size(), position, false);
    env_map_->vis();
    ROS_INFO_STREAM("env cloud num: " << cloud_env.points.size());
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "gen_occmap");
    ros::NodeHandle nh("~");

    tgt_map_ = std::make_shared<airgrasp::SDFMap>();
    tgt_map_->initMap(nh);
    env_map_ = std::make_shared<airgrasp::SDFMap>();
    env_map_->initMap(nh);

    double res;
    Eigen::Vector3d map_size, map_center;
    nh.param("tgt_map/resolution", res, -1.0);
    nh.param("tgt_map/map_size_x", map_size(0), -1.0);
    nh.param("tgt_map/map_size_y", map_size(1), -1.0);
    nh.param("tgt_map/map_size_z", map_size(2), -1.0);
    nh.param("tgt_map/map_center_x", map_center(0), -1.0);
    nh.param("tgt_map/map_center_y", map_center(1), -1.0);
    nh.param("tgt_map/map_center_z", map_center(2), -1.0);
    tgt_map_->setMapParam(res, map_size, map_center, "tgt_map");
    INFO_MSG("tgt_map initialized");

    nh.param("env_map/resolution", res, -1.0);
    nh.param("env_map/map_size_x", map_size(0), -1.0);
    nh.param("env_map/map_size_y", map_size(1), -1.0);
    nh.param("env_map/map_size_z", map_size(2), -1.0);
    nh.param("env_map/map_center_x", map_center(0), -1.0);
    nh.param("env_map/map_center_y", map_center(1), -1.0);
    nh.param("env_map/map_center_z", map_center(2), -1.0);    
    env_map_->setMapParam(res, map_size, map_center, "env_map");
    INFO_MSG("env_map initialized");

    std::string syncframe_topic;
    nh.param("syncframe_topic", syncframe_topic, std::string("/sync_frame_pcl"));
    ros::Subscriber sub = nh.subscribe(syncframe_topic, 10, syncFrameCallback);

    ros::spin();

    return 0;
}