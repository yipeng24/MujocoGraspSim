#include <ros/ros.h>
#include "plan_env/sdf_map.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <Eigen/Dense>
#include <iomanip>
#include <vector>
#include <limits>
#include <cmath>

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

void feedOdomMsg(nav_msgs::Odometry& odom_msg, const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation) {
    odom_msg.header.frame_id = "world";
    odom_msg.header.stamp = ros::Time::now();
    odom_msg.pose.pose.position.x = position.x();
    odom_msg.pose.pose.position.y = position.y();
    odom_msg.pose.pose.position.z = position.z();
    odom_msg.pose.pose.orientation.x = orientation.x();
    odom_msg.pose.pose.orientation.y = orientation.y();
    odom_msg.pose.pose.orientation.z = orientation.z();
    odom_msg.pose.pose.orientation.w = orientation.w();
}


// 读取相机位姿
bool readCameraPose(const std::string &file_path, Eigen::Vector3d &pos, Eigen::Quaterniond &ori) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "❌ 无法打开文件：" << file_path << std::endl;
        return false;
    }

    std::string line;

    // 读取 position
    if (std::getline(file, line)) {
        std::istringstream posStream(line);
        posStream >> pos.x() >> pos.y() >> pos.z();
    } else {
        std::cerr << "❌ 读取 position 失败！" << std::endl;
        return false;
    }

    // 读取 orientation
    if (std::getline(file, line)) {
        std::istringstream oriStream(line);
        oriStream >> ori.w() >> ori.x() >> ori.y() >> ori.z();
    } else {
        std::cerr << "❌ 读取 orientation 失败！" << std::endl;
        return false;
    }

    file.close();
    return true;
}

// 定义一个函数来写入两个 Eigen::Vector3i 到文件
void writeVectorsToFile(const std::string& filename, const Eigen::Vector3i& vec1, const Eigen::Vector3i& vec2) {
    std::ofstream outFile(filename);
    if (outFile.is_open()) {
        outFile << vec1.transpose() << "\n";  // 写入第一行
        outFile << vec2.transpose() << "\n";  // 写入第二行
        outFile.close();
        std::cout << "Vectors written to " << filename << std::endl;
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}


// 读取二值 mask 图像
bool loadMaskImage(const std::string& mask_path, cv::Mat& mask) {
    mask = cv::imread(mask_path, cv::IMREAD_GRAYSCALE); // 读取为灰度图

    if (mask.empty()) {
        std::cerr << "Error: Could not load mask image from " << mask_path << std::endl;
        return false;
    }

    // 将 mask 转换为布尔类型（0 变为 false，非 0 变为 true）
    mask = mask > 0;

    return true;
}


bool loadDepthImage(const std::string& path, cv::Mat& depth_real, bool add_noise = false) {
    cv::Mat depth_raw = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (depth_raw.empty()) {
        ROS_ERROR("Failed to load depth image: %s", path.c_str());
        return false;
    }

    // **映射回 0-5m**
    depth_raw.convertTo(depth_real, CV_32F, 5.0 / 255.0);

    // **是否添加噪声**
    if (add_noise) {
        cv::Mat noise(depth_real.size(), CV_32F);
        std::random_device rd;
        std::mt19937 gen(rd());

        for (int y = 0; y < depth_real.rows; ++y) {
            for (int x = 0; x < depth_real.cols; ++x) {
                float depth_value = depth_real.at<float>(y, x);
                if (depth_value <= 0.3f) continue;  // 忽略无效深度

                // **D435 的噪声模型**
                float sigma = 0.001f * (depth_value * depth_value);
                std::normal_distribution<float> noise_dist(0.0f, sigma);
                float noise_value = noise_dist(gen);

                // **添加噪声并确保深度范围**
                depth_real.at<float>(y, x) = std::max(0.0f, depth_value + noise_value);
            }
        }
    }

    return true;
}

// **深度图转换为 3D 点云，每个点携带 mask 信息**
pcl::PointCloud<pcl::PointXYZI>::Ptr depthToPointCloud(
        const cv::Mat& depth, 
        const cv::Mat& mask,
        const Eigen::Matrix3d& K, 
        const Eigen::Matrix3d& R, 
        const Eigen::Vector3d& T) {

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());

    int width = depth.cols;
    int height = depth.rows;
    
    Eigen::Matrix3d K_inv = K.inverse(); // 计算内参矩阵的逆

    for (int v = 0; v < height; ++v) {
        for (int u = 0; u < width; ++u) {
            float d = depth.at<float>(v, u);
            if (d <= 0.3) continue; // 深度过小，跳过无效点

            Eigen::Vector3d cam_point = K_inv * Eigen::Vector3d(u * d, v * d, d);
            Eigen::Vector3d world_point = R * cam_point + T; // 变换到世界坐标系
            
            pcl::PointXYZI point;
            point.x = world_point.x();
            point.y = world_point.y();
            point.z = world_point.z();
            point.intensity = static_cast<float>(mask.at<uint8_t>(v, u)); // 存入 mask 值（0 or 1）

            cloud->points.push_back(point);
        }
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = false;
    return cloud;
}

// **将深度图转换为 3D 点云**
pcl::PointCloud<pcl::PointXYZ>::Ptr depthToPointCloud(const cv::Mat& depth, 
                                                      const Eigen::Matrix3d& K, 
                                                      const Eigen::Matrix3d& R, 
                                                      const Eigen::Vector3d& T) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());

    int width = depth.cols;
    int height = depth.rows;

    Eigen::Matrix3d K_inv = K.inverse(); // 计算 K 的逆矩阵

    for (int v = 0; v < height; ++v) {
        for (int u = 0; u < width; ++u) {
            float d = depth.at<float>(v, u);
            if (d <= 0.3) continue; // 跳过无效深度点

            Eigen::Vector3d cam_point = K_inv * Eigen::Vector3d(u * d, v * d, d);
            Eigen::Vector3d world_point = R * cam_point + T; // 变换到世界坐标系
            
            cloud->points.emplace_back(world_point.x(), world_point.y(), world_point.z());
        }
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = false;
    return cloud;
}

// **ROS 话题发布（支持 PointXYZ & PointXYZI）**
template <typename PointT>
void publishPointCloud(ros::Publisher& pub, typename pcl::PointCloud<PointT>::Ptr cloud) {
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*cloud, output);
    output.header.frame_id = "world";  // 设置全局坐标系
    output.header.stamp = ros::Time::now();
    pub.publish(output);
}

// **沿 Y 轴 +90°，沿 Z 轴 -90°**
void transformCameraPose(Eigen::Vector3d& position, Eigen::Quaterniond& orientation) {
    Eigen::Matrix3d R_cam2body;
    R_cam2body << 0.0,  0.0,  1.0,
             -1.0,  0.0,  0.0,
              0.0, -1.0,  0.0;

    // **变换方向**
    orientation =  orientation*Eigen::Quaterniond(R_cam2body);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr extractAndConvert(
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_mask) {
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_obj(new pcl::PointCloud<pcl::PointXYZ>);

    for (const auto& point : cloud_mask->points) {
        if (point.intensity > 1.0) {
            cloud_obj->points.emplace_back(point.x, point.y, point.z);
        }
    }

    cloud_obj->width = cloud_obj->points.size();
    cloud_obj->height = 1;
    cloud_obj->is_dense = true;

    return cloud_obj;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr farthestPointSampling(
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud, int num_samples) {

    int num_points = input_cloud->size();
    if (num_samples >= num_points) return input_cloud; // 不需要采样

    pcl::PointCloud<pcl::PointXYZ>::Ptr sampled_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<bool> selected(num_points, false);
    std::vector<float> min_distances(num_points, std::numeric_limits<float>::max());

    int first_idx = rand() % num_points;
    sampled_cloud->points.push_back(input_cloud->points[first_idx]);
    selected[first_idx] = true;

    for (int i = 1; i < num_samples; ++i) {
        float max_dist = -1;
        int max_idx = -1;

        for (int j = 0; j < num_points; ++j) {
            if (selected[j]) continue;
            float dist = std::sqrt(
                std::pow(input_cloud->points[j].x - sampled_cloud->points.back().x, 2) +
                std::pow(input_cloud->points[j].y - sampled_cloud->points.back().y, 2) +
                std::pow(input_cloud->points[j].z - sampled_cloud->points.back().z, 2)
            );
            min_distances[j] = std::min(min_distances[j], dist);
            if (min_distances[j] > max_dist) {
                max_dist = min_distances[j];
                max_idx = j;
            }
        }

        sampled_cloud->points.push_back(input_cloud->points[max_idx]);
        selected[max_idx] = true;
    }

    sampled_cloud->width = num_samples;
    sampled_cloud->height = 1;
    sampled_cloud->is_dense = true;
    return sampled_cloud;
}


std::shared_ptr<fast_planner::SDFMap> tgt_map_, env_map_;

int main(int argc, char **argv)
{
    ros::init(argc, argv, "gen_occmap");
    ros::NodeHandle nh("~");

    ros::Publisher cloud_depth_pub = nh.advertise<sensor_msgs::PointCloud2>("pcl_depth", 1);
    ros::Publisher cloud_mask_pub = nh.advertise<sensor_msgs::PointCloud2>("pcl_mask", 1);
    ros::Publisher cloud_obj_sampled_pub = nh.advertise<sensor_msgs::PointCloud2>("pcl_obj_sampled", 1);
    ros::Publisher odom_pub = nh.advertise<nav_msgs::Odometry>("camera_pose", 10);

    ros::Publisher cloud_env_save_pub = nh.advertise<sensor_msgs::PointCloud2>("pcl_env_save", 1);

    tgt_map_ = std::make_shared<fast_planner::SDFMap>();
    tgt_map_->initMap(nh);
    env_map_ = std::make_shared<fast_planner::SDFMap>();
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

    // std::string base_path = "/home/jr/proj/dataset/sensor_data/cup_0049/2025-02-13_20-20-52/drop_0/drone_0/";
    std::string base_path = "/home/jr/proj/dataset/sensor_data/cup_0049/2025-02-13_20-11-48/drop_0/drone_0/";

    // record voxel size
    std::string voxel_size_path = base_path + "grid/voxel_num.txt";
    writeVectorsToFile(voxel_size_path, tgt_map_->get3DVoxelNum(), env_map_->get3DVoxelNum());
    INFO_MSG("save voxel size to " << voxel_size_path);

    Eigen::Matrix3d K,K_inv;
    K << 612.4178466796875, 0.0, 309.72296142578125, 
        0.0, 612.362060546875, 245.35870361328125, 
        0.0, 0.0, 1.0;
    K_inv = K.inverse();

    pcl::PointCloud<pcl::PointXYZ>::Ptr previous_sampled(new pcl::PointCloud<pcl::PointXYZ>);

    ros::Rate loop_rate(3);  // 1 Hz
    int idx = 5;
    while (ros::ok()) {
        std::ostringstream oss;
        oss << std::setw(3) << std::setfill('0') << idx;  
        std::string idx_str = oss.str();
        std::string file_path = base_path + "d_0_cam_pose_f_" + idx_str + ".txt";

        INFO_MSG("============ frame " << idx_str << " ============");
        INFO_MSG("============ frame " << idx_str << " ============");
        INFO_MSG("============ frame " << idx_str << " ============");
        std::cout << "File Path: " << file_path << std::endl;

        //! 相机位姿
        Eigen::Vector3d position;
        Eigen::Quaterniond orientation;

        if (readCameraPose(file_path, position, orientation)) {
            transformCameraPose(position, orientation);
            std::cout << "Position: " << position.transpose() << std::endl;
            std::cout << "Orientation (x, y, z, w): " << orientation.coeffs().transpose() << std::endl;
            
            nav_msgs::Odometry odom_msg;
            feedOdomMsg(odom_msg, position, orientation);
            odom_pub.publish(odom_msg);
            INFO_MSG("publish camera pose done");
        } else {
            INFO_MSG_RED("get camera pose failed");
        }

        //! 深度图
        INFO_MSG("load depth image");
        file_path = base_path + "frame_" + idx_str + "_depth.png";
        cv::Mat depth;
        if (!loadDepthImage(file_path, depth, true)){
            idx++;
            continue;
        }

        Eigen::Matrix3d rot = orientation.toRotationMatrix();
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_all;
        cloud_all = depthToPointCloud(depth, K, rot, position);
        publishPointCloud<pcl::PointXYZ>(cloud_depth_pub, cloud_all);

        //! mask
        INFO_MSG("load mask image");
        cv::Mat mask;
        file_path = base_path + "frame_" + idx_str + "_mask.png";
        if (loadMaskImage(file_path, mask)) {
            std::cout << "Loaded mask image successfully: " << file_path << std::endl;
            std::cout << "Mask size: " << mask.cols << " x " << mask.rows << std::endl;
        } else {
            std::cerr << "Failed to load mask image!" << std::endl;
            continue;
        }

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_mask;
        cloud_mask = depthToPointCloud(depth, mask, K, rot, position);
        publishPointCloud<pcl::PointXYZI>(cloud_mask_pub, cloud_mask);

        //! feed pcl to sdf_map
        file_path = base_path + "grid/frame_" + idx_str + "_obj_grid.pcd";
        tgt_map_->inputPointCloud(*cloud_mask, cloud_mask->points.size(),position);
        tgt_map_->publishMapInfo();
        pcl::PointCloud<pcl::PointXYZI>::Ptr grid_obj = tgt_map_->getPointCloudXYZI();
        pcl::io::savePCDFileASCII(file_path, *grid_obj); // 保存为 PCD 文件
        INFO_MSG("save target cloud to "<< file_path);

        file_path = base_path + "grid/frame_" + idx_str + "_env_grid.pcd";
        env_map_->inputPointCloud(*cloud_mask, cloud_mask->points.size(),position);
        env_map_->publishMapInfo();
        pcl::PointCloud<pcl::PointXYZI>::Ptr grid_env = env_map_->getPointCloudXYZI();
        pcl::io::savePCDFileASCII(file_path, *grid_env); // 保存为 PCD 文件
        // publishPointCloud<pcl::PointXYZI>(cloud_env_save_pub, cloud_env);
        // INFO_MSG("save env cloud to "<< file_path);

        //! 采样
        int num_samples = 256;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_obj = extractAndConvert(cloud_mask);
        pcl::PointCloud<pcl::PointXYZ>::Ptr flt_cloud_obj(new pcl::PointCloud<pcl::PointXYZ>);
        if (previous_sampled->size() > 0) {
            // 数据集时间戳有时候有问题 
            // 过滤离群点
            for (const auto& pt : previous_sampled->points){ 
                Eigen::Vector3d pos(pt.x, pt.y, pt.z);
                // if (tgt_map_->isUnknown(pos) || tgt_map_->isOccupied(pos)) {
                if (tgt_map_->isObject(pos)) {
                    flt_cloud_obj->points.push_back(pt);
                }
            }
            flt_cloud_obj->width = flt_cloud_obj->points.size();
            flt_cloud_obj->height = 1;
            flt_cloud_obj->is_dense = true;

            cloud_obj->points.insert(cloud_obj->points.end(), flt_cloud_obj->points.begin(), flt_cloud_obj->points.end());
        }
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_obj_sampled = farthestPointSampling(cloud_obj, num_samples);
        publishPointCloud<pcl::PointXYZ>(cloud_obj_sampled_pub, cloud_obj_sampled);
        file_path = base_path + "grid/frame_" + idx_str + "_obj_sampled_pcl.pcd";
        pcl::io::savePCDFileASCII(file_path, *cloud_obj_sampled); // 保存为 PCD 文件
        
        previous_sampled = cloud_obj_sampled;

        ros::spinOnce();
        loop_rate.sleep();

        idx++;
    }

    return 0;
}