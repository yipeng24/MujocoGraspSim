#pragma once
#include <ros/ros.h>
#include <stdio.h>
#include <dirent.h>

#include <mutex>
#include <Eigen/Dense>

#include "rosbag_direct_write/direct_bag.h"

#include "geometry_msgs/PoseStamped.h"
#include <visualization_msgs/Marker.h>
#include <stdlib.h>


namespace rosbag_tool{
enum Color { white,
             red,
             green,
             blue,
             yellow,
             greenblue };
class Bag{
 private:
    std::shared_ptr<rosbag_direct_write::DirectBag> rosbag_;
    std::mutex mutex_rosbag_;
    std::string bag_path_;
    int bag_id_;

 public:
    Bag(const std::string bag_name){
        bag_path_ = "./bag/" + bag_name + ".bag";
        std::string folderpath  = "./bag";
        std::string command = "mkdir -p " + folderpath;
        system(command.c_str());
        rosbag_ = std::make_shared<rosbag_direct_write::DirectBag>(bag_path_, false);
        if (!rosbag_->is_open()){
            std::cerr << "Fail to open rosbag\n" << std::endl;
        }
    }

    int get_bag_id(){
        return bag_id_;
    }

    std::string get_bag_path(){
        return bag_path_;
    }

    void close(){
        rosbag_->close();
        std::cout << "rosbag closed. -> " << bag_path_ << std::endl;
    }

 private:
    int calculate_bag_id(){
        DIR* d = opendir("./bag");
        if (d == NULL){
            std::cerr << "Fail to calculate bag id." << std::endl;
        }
        struct dirent* entry;
        int max_id = 0;
        while ((entry = readdir(d)) != NULL)
        {
            std::string name = entry->d_name;
            int i = 0;
            int len = 0;
            while (i < name.size()){
                if (name[i] >= '0' && name[i] <= '9'){
                    len++;
                }
                i++;
            }
            if (len == 0) continue;
            std::string s0 = name.substr(0, len);
            int num = 0;
            std::stringstream s1(s0);
            s1 >> num;
            if (num > max_id) max_id = num;
        }
        closedir(d);
        return max_id + 1;
    }

    void setMarkerColor(visualization_msgs::Marker& marker,
                        Color color = blue,
                        double a = 1) {
        marker.color.a = a;
        switch (color) {
        case white:
            marker.color.r = 1;
            marker.color.g = 1;
            marker.color.b = 1;
            break;
        case red:
            marker.color.r = 1;
            marker.color.g = 0;
            marker.color.b = 0;
            break;
        case green:
            marker.color.r = 0;
            marker.color.g = 1;
            marker.color.b = 0;
            break;
        case blue:
            marker.color.r = 0;
            marker.color.g = 0;
            marker.color.b = 1;
            break;
        case yellow:
            marker.color.r = 1;
            marker.color.g = 1;
            marker.color.b = 0;
            break;
        case greenblue:
            marker.color.r = 0;
            marker.color.g = 1;
            marker.color.b = 1;
            break;
        }
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
    template <class ROTATION>
    void setMarkerPose(visualization_msgs::Marker& marker,
                        const double& x,
                        const double& y,
                        const double& z,
                        const ROTATION& R) {
        marker.pose.position.x = x;
        marker.pose.position.y = y;
        marker.pose.position.z = z;
        Eigen::Quaterniond r(R);
        marker.pose.orientation.w = r.w();
        marker.pose.orientation.x = r.x();
        marker.pose.orientation.y = r.y();
        marker.pose.orientation.z = r.z();
    }



 public:

    void write_posestamp(const std::string topic, const Eigen::Vector3d& pos, const Eigen::Quaterniond& q,
                         const ros::Time& time, const char* frame_id = "world")
    {
        if (!rosbag_->is_open()) return;
        geometry_msgs::PoseStamped tmp_pose;
        tmp_pose.header.frame_id = frame_id;
        tmp_pose.header.stamp = time;
        tmp_pose.pose.position.x = pos.x();
        tmp_pose.pose.position.y = pos.y();
        tmp_pose.pose.position.z = pos.z();
        tmp_pose.pose.orientation.w = q.w();
        tmp_pose.pose.orientation.x = q.x();
        tmp_pose.pose.orientation.y = q.y();
        tmp_pose.pose.orientation.z = q.z();

        std::lock_guard<std::mutex> lk(mutex_rosbag_);
        rosbag_->write(topic, time, tmp_pose);
    }
};



}

