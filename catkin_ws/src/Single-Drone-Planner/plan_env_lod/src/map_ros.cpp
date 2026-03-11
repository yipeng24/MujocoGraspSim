#include <plan_env_lod/sdf_map.h>
#include <plan_env_lod/map_ros.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <visualization_msgs/Marker.h>

#include <fstream>

namespace airgrasp {
MapROS::MapROS() {
}

MapROS::~MapROS() {
}

void MapROS::setMap(SDFMap* map) {
  this->map_ = map;
}

void MapROS::setMapName(const std::string& map_name) {
  this->map_name_ = map_name;
}

void MapROS::init() {
  node_.param("map_ros/fx", cam_param_.fx, -1.0);
  node_.param("map_ros/fy", cam_param_.fy, -1.0);
  node_.param("map_ros/cx", cam_param_.cx, -1.0);
  node_.param("map_ros/cy", cam_param_.cy, -1.0);
  if(cam_param_.fx < 0 || cam_param_.fy < 0 || cam_param_.cx < 0 || cam_param_.cy < 0) {
    ROS_ERROR("Camera parameters not set");
    ROS_ERROR("Camera parameters not set");
    ROS_ERROR("Camera parameters not set");
  }

  node_.param("map_ros/depth_filter_maxdist", cam_param_.max_range, -1.0);
  node_.param("map_ros/depth_filter_mindist", cam_param_.min_range, -1.0);
  node_.param("map_ros/depth_filter_margin", depth_filter_margin_, -1);
  node_.param("map_ros/k_depth_scaling_factor", k_depth_scaling_factor_, -1.0);
  node_.param("map_ros/skip_pixel", skip_pixel_, -1);

  node_.param("map_ros/esdf_slice_height", esdf_slice_height_, -0.1);
  node_.param("map_ros/visualization_truncate_height", visualization_truncate_height_, -0.1);
  node_.param("map_ros/visualization_truncate_low", visualization_truncate_low_, -0.1);
  node_.param("map_ros/show_occ_time", show_occ_time_, false);
  node_.param("map_ros/show_esdf_time", show_esdf_time_, false);
  node_.param("map_ros/show_all_map", show_all_map_, false);
  node_.param("map_ros/frame_id", frame_id_, string("world"));

  node_.param("map_ros/input_cam_pose", input_cam_pose_, true);

  // std::cout << "-------------------------------------------" << std::endl;
  // std::cout << "-------------------------------------------" << std::endl;
  // std::cout << "fx: " << cam_param_.fx << std::endl;
  // std::cout << "fy: " << cam_param_.fy << std::endl;

  cam2inputPose_ << 0.0, 0.0, 1.0, 0.06,
                   -1.0, 0.0, 0.0, 0.0,
                    0.0, -1.0, 0.0, 0.1,
                    0.0, 0.0, 0.0, 1.0;
  cam2inputPose_ = input_cam_pose_ ? cam2inputPose_ : Eigen::Matrix4d::Identity();
  // std::cout << "----- cam2inputPose -----" << std::endl;
  // std::cout << cam2inputPose_ << std::endl;

  // proj_points_.resize(640 * 480 / (skip_pixel_ * skip_pixel_));
  // point_cloud_.points.resize(640 * 480 / (skip_pixel_ * skip_pixel_));
  // proj_points_.reserve(640 * 480 / map_->mp_->skip_pixel_ / map_->mp_->skip_pixel_);
  proj_points_cnt = 0;

  local_updated_ = false;
  esdf_need_update_ = false;
  have_first_depth_ = false;
  fuse_time_ = 0.0;
  esdf_time_ = 0.0;
  max_fuse_time_ = 0.0;
  max_esdf_time_ = 0.0;
  fuse_num_ = 0;
  esdf_num_ = 0;
  depth_image_.reset(new cv::Mat);

  rand_noise_ = normal_distribution<double>(0, 0.1);
  random_device rd;
  eng_ = default_random_engine(rd());

  registROS();
}

// TODO TODO TODO
void MapROS::registROS() {

  // esdf_timer_ = node_.createTimer(ros::Duration(0.05), &MapROS::updateESDFCallback, this);
  vis_timer_ = node_.createTimer(ros::Duration(0.1), &MapROS::visCallback, this);

  // map_all_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/occupancy_all", 10);
  // map_local_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/occupancy_local", 10);
  // map_local_inflate_pub_ =
  //     node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/occupancy_local_inflate", 10);
  // unknown_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/unknown", 10);
  // esdf_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/esdf", 10);
  // update_range_pub_ = node_.advertise<visualization_msgs::Marker>("/sdf_map/update_range", 10);
  // depth_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/sdf_map/depth_cloud", 10);

  // grid_CR_img_pub_ = node_.advertise<sensor_msgs::Image>("/sdf_map/grid_CR_img", 10);

  // depth_sub_.reset(new message_filters::Subscriber<sensor_msgs::Image>(node_, "/map_ros/depth", 1, ros::TransportHints().tcpNoDelay()));
  // cloud_sub_.reset(
  //     new message_filters::Subscriber<sensor_msgs::PointCloud2>(node_, "/map_ros/cloud", 50));
  // pose_sub_.reset(
  //     new message_filters::Subscriber<nav_msgs::Odometry>(node_, "/map_ros/pose", 25));

  // sync_image_pose_.reset(new message_filters::Synchronizer<MapROS::SyncPolicyImagePose>(
  //     MapROS::SyncPolicyImagePose(100), *depth_sub_, *pose_sub_));
  // sync_image_pose_->registerCallback(boost::bind(&MapROS::depthPoseCallback, this, _1, _2));
  // sync_cloud_pose_.reset(new message_filters::Synchronizer<MapROS::SyncPolicyCloudPose>(
  //     MapROS::SyncPolicyCloudPose(100), *cloud_sub_, *pose_sub_));
  // sync_cloud_pose_->registerCallback(boost::bind(&MapROS::cloudPoseCallback, this, _1, _2));

  map_start_time_ = ros::Time::now();
}

void MapROS::move_base_callback(const geometry_msgs::PoseStamped::ConstPtr &msg){
  Eigen::Vector3d goal;
  goal(0) = msg->pose.position.x;
  goal(1) = msg->pose.position.y;
  goal(2) = msg->pose.position.z;
  double yaw;
  yaw = tf::getYaw(msg->pose.orientation);
  map_->debugCheckRayValid(goal, yaw);
}


void MapROS::visCallback(const ros::TimerEvent& e) {
  publishMapOcc();
  publishMapInf();
  // publishESDF();
  // publishLoss();
  // publishUnknown();
}

void MapROS::updateESDFCallback(const ros::TimerEvent& /*event*/) {
  if (!esdf_need_update_ or benchmark_mode_) return;
  // ROS_ERROR("in updateESDFCallback");
  auto t1 = ros::Time::now();

  map_->updateESDF3d();
  esdf_need_update_ = false;

  auto t2 = ros::Time::now();
  esdf_time_ += (t2 - t1).toSec();
  max_esdf_time_ = max(max_esdf_time_, (t2 - t1).toSec());
  esdf_num_++;
  if (show_esdf_time_)
    ROS_WARN("ESDF t: cur: %lf, avg: %lf, max: %lf", (t2 - t1).toSec(), esdf_time_ / esdf_num_,
             max_esdf_time_);

  // std::cout << "out of updateESDFCallback" << std::endl;
}

void MapROS::depthPoseCallback(const sensor_msgs::ImageConstPtr& img,
                               const nav_msgs::OdometryConstPtr& pose) {
  Eigen::Quaterniond body_q(pose->pose.pose.orientation.w, pose->pose.pose.orientation.x,
                            pose->pose.pose.orientation.y, pose->pose.pose.orientation.z);

  if(benchmark_mode_) return;

  // ROS_ERROR("In depthPoseCallback");

  // sub body pose
  Eigen::Matrix4d body2world,cam_T;
  body2world.block<3, 3>(0, 0) = body_q.toRotationMatrix();
  body2world(0, 3) = pose->pose.pose.position.x;
  body2world(1, 3) = pose->pose.pose.position.y;
  body2world(2, 3) = pose->pose.pose.position.z;
  body2world(3, 3) = 1.0;

  cam_T = body2world * cam2inputPose_;
  camera_pos_(0) = cam_T(0, 3);
  camera_pos_(1) = cam_T(1, 3);
  camera_pos_(2) = cam_T(2, 3);
  camera_q_ = Eigen::Quaterniond(cam_T.block<3, 3>(0, 0));

  // sub camera pose
  // camera_pos_(0) = pose->pose.pose.position.x;
  // camera_pos_(1) = pose->pose.pose.position.y;
  // camera_pos_(2) = pose->pose.pose.position.z;
  // camera_q_ = body_q;

  if (!map_->isInMap(camera_pos_))  // exceed mapped region
    return;

  // ROS_ERROR("1 in depthPoseCallback");

  cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img, img->encoding);
  // std::cout << "encoding: " << img->encoding << std::endl;
  // if (img->encoding == sensor_msgs::image_encodings::TYPE_32FC1)
  //   (cv_ptr->image).convertTo(cv_ptr->image, CV_16UC1, k_depth_scaling_factor_);
  cv_ptr->image.copyTo(*depth_image_);

  auto t1 = ros::Time::now();


// 去除第一帧周围
// 作图
  if(!have_first_depth_)
  {
    map_->setCamearPose2free(camera_pos_);
    have_first_depth_ = true;
  }

  //  ROS_ERROR("2 in depthPoseCallback");
  
  // generate point cloud, update map
  proessDepthImage();

  // ROS_ERROR("2.2 in depthPoseCallback");

  map_->inputPointCloud(point_cloud_, proj_points_cnt, camera_pos_);

  // ROS_ERROR("2.5 in depthPoseCallback");

  if (local_updated_) {
    map_->clearAndInflateLocalMap();
    esdf_need_update_ = true;
    local_updated_ = false;
  }

  // ROS_ERROR("3 in depthPoseCallback");

  auto t2 = ros::Time::now();
  fuse_time_ += (t2 - t1).toSec();
  max_fuse_time_ = max(max_fuse_time_, (t2 - t1).toSec());
  fuse_num_ += 1;
  if (show_occ_time_)
    ROS_WARN("Fusion t: cur: %lf, avg: %lf, max: %lf", (t2 - t1).toSec(), fuse_time_ / fuse_num_,
             max_fuse_time_);

  // ROS_ERROR("rnd in depthPoseCallback");
}

// void MapROS::cloudPoseCallback(const sensor_msgs::PointCloud2ConstPtr& msg,
//                                const geometry_msgs::PoseStampedConstPtr& pose) {
//   camera_pos_(0) = pose->pose.position.x;
//   camera_pos_(1) = pose->pose.position.y;
//   camera_pos_(2) = pose->pose.position.z;
//   camera_q_ = Eigen::Quaterniond(pose->pose.orientation.w, pose->pose.orientation.x,
//                                  pose->pose.orientation.y, pose->pose.orientation.z);
//   pcl::PointCloud<pcl::PointXYZ> cloud;
//   pcl::fromROSMsg(*msg, cloud);
//   int num = cloud.points.size();

//   map_->inputPointCloud(cloud, num, camera_pos_);

//   if (local_updated_) {
//     map_->clearAndInflateLocalMap();
//     esdf_need_update_ = true;
//     local_updated_ = false;
//   }
// }

void MapROS::proessDepthImage() {

  // ROS_ERROR("In proessDepthImage");

  proj_points_cnt = 0;
  point_cloud_.points.clear();

  // float* row_ptr;
  int cols = depth_image_->cols;
  int rows = depth_image_->rows;
  // std::cout << "cols: " << cols << ", rows: " << rows << std::endl;
  double depth;
  Eigen::Matrix3d camera_r = camera_q_.toRotationMatrix();
  Eigen::Vector3d pt_cur, pt_world;
  const double inv_factor = 1.0 / k_depth_scaling_factor_;
  // std::cout << "inv_factor: " << inv_factor << std::endl;
  // std::cout << "depth(0,0)" << depth_image_->at<float>(0, 0)* inv_factor << std::endl;

  // std::cout << "cam_pos: " << camera_pos_ << std::endl;
  // std::cout << "cam_q: " << camera_q_.coeffs() << std::endl;

  for (int v = depth_filter_margin_; v < rows - depth_filter_margin_; v += skip_pixel_) {
    // row_ptr = depth_image_->ptr<float>(v) + depth_filter_margin_;
    for (int u = depth_filter_margin_; u < cols - depth_filter_margin_; u += skip_pixel_) {

      // depth = (*row_ptr) * inv_factor;

      double depth = depth_image_->at<float>(v, u) * inv_factor;
      // double depth = (depth_image_->at<float>(v, u)) * inv_factor;

      // std::cout << "depth: " << depth << std::endl;

      // row_ptr = row_ptr + skip_pixel_;

      // // filter depth
      // if (depth > 0.01)
      //   depth += rand_noise_(eng_);

      bool out_of_range = false;

      // TODO: simplify the logic here
      if (depth < DBL_EPSILON /*无穷远*/ || depth > cam_param_.max_range)
      {
        out_of_range = true;
        depth = cam_param_.max_range;
      }
      else if (depth < cam_param_.min_range || std::isnan(depth))
        continue;

      pt_cur(0) = (u - cam_param_.cx) * depth / cam_param_.fx;
      pt_cur(1) = (v - cam_param_.cy) * depth / cam_param_.fy;
      pt_cur(2) = depth;
      pt_world = camera_r * pt_cur + camera_pos_;
      
      pt_world = out_of_range
                ? camera_pos_ + (pt_world-camera_pos_).normalized() * (cam_param_.max_range+1e-3) 
                : pt_world;

      pcl::PointXYZ pcl;
      pcl.x = pt_world[0];
      pcl.y = pt_world[1];
      pcl.z = pt_world[2];
      point_cloud_.push_back(pcl);
      proj_points_cnt++;

      // std::cout << "proj_points_cnt: " << proj_points_cnt << std::endl;

      // auto& pt = point_cloud_.points[proj_points_cnt++];
      // pt.x = pt_world[0];
      // pt.y = pt_world[1];
      // pt.z = pt_world[2];
    }
  }
  // std::cout << "out of loop" << std::endl;

  // ROS_ERROR("end proessDepthImage");

  publishDepth();
}


void MapROS::publishMapInf() {
  std::vector<std::pair<Eigen::Vector3d, double>> pcl_i;
  for (int x = 0; x < map_->mp_->map_voxel_num_(0); ++x)
    for (int y = 0; y < map_->mp_->map_voxel_num_(1); ++y)
      for (int z = 0; z < map_->mp_->map_voxel_num_(2); ++z) {
        if(map_->md_->occupancy_buffer_inflate_[map_->toAddress(x, y, z)] == 1){
          Eigen::Vector3d pos;
          map_->indexToPos(Eigen::Vector3i(x, y, z), pos);
          if (pos(2) > visualization_truncate_height_) continue;
          if (pos(2) < visualization_truncate_low_) continue;
          pcl_i.push_back(std::make_pair(pos, 1.0));
        }
      }
  vis_ptr_->visualize_pointcloud_itensity(pcl_i, "sdf_map/inf_occ");

  // std::vector<std::pair<Eigen::Vector3d, double>> pcl_i;
  // for(double x = map_->mp_->map_min_boundary_(0); x < map_->mp_->map_max_boundary_(0); x += map_->mp_->resolution_)
  //   for(double y = map_->mp_->map_min_boundary_(1); y < map_->mp_->map_max_boundary_(1); y += map_->mp_->resolution_)
  //     for(double z = map_->mp_->map_min_boundary_(2); z < map_->mp_->map_max_boundary_(2); z += map_->mp_->resolution_){
  //       if(map_->isInfOccupied(Eigen::Vector3d(x, y, z))){
  //         if (z > visualization_truncate_height_) continue;
  //         if (z < visualization_truncate_low_) continue;
  //         pcl_i.push_back(std::make_pair(Eigen::Vector3d(x, y, z), 0));
  //       }
  //     }
  // vis_ptr_->visualize_pointcloud_itensity(pcl_i, "sdf_map/inf_occ");
}

void MapROS::publishUnknown() {
  std::vector<std::pair<Eigen::Vector3d, double>> pcl_i;
  for (int x = 0; x < map_->mp_->map_voxel_num_(0); ++x)
    for (int y = 0; y < map_->mp_->map_voxel_num_(1); ++y)
      for (int z = 0; z < map_->mp_->map_voxel_num_(2); ++z)
        if(map_->isUnknown(Eigen::Vector3i(x, y, z)))
        {
          Eigen::Vector3d pt;
          map_->indexToPos(Eigen::Vector3i(x, y, z), pt);
          pcl_i.push_back(std::make_pair(pt, 1.0));
        }

  vis_ptr_->visualize_pointcloud_itensity(pcl_i, "sdf_map/unknown3d");
}

void MapROS::publishInfo() {
  std::vector<std::pair<Eigen::Vector3d, double>> pcl_i;
  for (int x = 0; x < map_->mp_->map_voxel_num_(0); ++x)
  for (int y = 0; y < map_->mp_->map_voxel_num_(1); ++y)
  for (int z = 0; z < map_->mp_->map_voxel_num_(2); ++z){

    Eigen::Vector3d pt;
    map_->indexToPos(Eigen::Vector3i(x, y, z), pt);

    if(map_->isUnknown(Eigen::Vector3i(x, y, z))){
      pcl_i.push_back(std::make_pair(pt, (double)VoxelStatus::UNKNOWN));
      continue;
    }

    if(map_->isOccupied(Eigen::Vector3i(x, y, z))){
      pcl_i.push_back(std::make_pair(pt, (double)VoxelStatus::OCCUPIED));
      continue;
    }
  }

  vis_ptr_->visualize_pointcloud_itensity(pcl_i, this->map_name_+"/info");
}

void MapROS::publishMapOcc() {
  std::vector<Eigen::Vector3d> pcl, pcl_gs;
  
  for (int x = 0; x < map_->mp_->map_voxel_num_(0); ++x)
    for (int y = 0; y < map_->mp_->map_voxel_num_(1); ++y)
      for (int z = 0; z < map_->mp_->map_voxel_num_(2); ++z) {
        if (map_->isOccupied(Eigen::Vector3i(x, y, z))) {
          Eigen::Vector3d pos;
          map_->indexToPos(Eigen::Vector3i(x, y, z), pos);
          // pos.z() = 0.1;
          if (pos(2) > visualization_truncate_height_) continue;
          if (pos(2) < visualization_truncate_low_) continue;
          pcl.push_back(pos);

          Eigen::Vector3d pos_gs = map_->index2mean(Eigen::Vector3i(x, y, z));


          if(std::isnan(pos_gs(0)) || std::isnan(pos_gs(1)) || std::isnan(pos_gs(2))){
            std::cout << "nan in pos_gs" << std::endl;
            std::cout << "nan in pos_gs" << std::endl;
            std::cout << "nan in pos_gs" << std::endl;
            std::cout << pos_gs.transpose() << std::endl;
          }

          pcl_gs.push_back(pos_gs);
        }
      }

  vis_ptr_->visualize_pointcloud(pcl, this->map_name_+"/occ");
  vis_ptr_->visualize_pointcloud(pcl_gs, this->map_name_+"/occ_gs");
}


void MapROS::publishLoss() {
  std::vector<std::pair<Eigen::Vector3d, double>> pcl_i;
  double max_z = -1;

  for (int x = 0; x < map_->mp_->map_voxel_num_(0); ++x)
    for (int y = 0; y < map_->mp_->map_voxel_num_(1); ++y)
      for (int z = 0; z < map_->mp_->map_voxel_num_(2); ++z) {

// debug
        // double pos_z = (z + 0.5) * map_->mp_->resolution_ + map_->mp_->map_origin_(2);
        // max_z = max(max_z, pos_z);

        if (map_->isOccupied(Eigen::Vector3i(x, y, z))) {
          Eigen::Vector3d pos;
          map_->indexToPos(Eigen::Vector3i(x, y, z), pos);

          // pos.z() = 0.1;
          // if (pos(2) > visualization_truncate_height_) continue;
          // if (pos(2) < visualization_truncate_low_) continue;

          pcl_i.push_back(std::make_pair(pos, 1.0));
        }
      }

// debug
  // std::cout << "[pubLoss]: max_z: " << max_z << std::endl;

  vis_ptr_->visualize_pointcloud_itensity(pcl_i, "sdf_map/loss");
}

void MapROS::publishDepth() {
  // ROS_ERROR("In publishDepth");
  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud;
  for (size_t i = 0; i < proj_points_cnt; ++i) {
    pt = point_cloud_.points[i];
    Eigen::Vector3d pos(pt.x, pt.y, pt.z);
    if((pos-camera_pos_).norm() > cam_param_.max_range - 1e-3)
      continue;
    cloud.push_back(point_cloud_.points[i]);
  }
  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  depth_pub_.publish(cloud_msg);
  // ROS_ERROR("end publishDepth");
}

// void MapROS::publishUpdateRange() {
//   Eigen::Vector3d esdf_min_pos, esdf_max_pos, cube_pos, cube_scale;
//   visualization_msgs::Marker mk;
//   map_->indexToPos(map_->md_->local_bound_min_, esdf_min_pos);
//   map_->indexToPos(map_->md_->local_bound_max_, esdf_max_pos);

//   cube_pos = 0.5 * (esdf_min_pos + esdf_max_pos);
//   cube_scale = esdf_max_pos - esdf_min_pos;
//   mk.header.frame_id = frame_id_;
//   mk.header.stamp = ros::Time::now();
//   mk.type = visualization_msgs::Marker::CUBE;
//   mk.action = visualization_msgs::Marker::ADD;
//   mk.id = 0;
//   mk.pose.position.x = cube_pos(0);
//   mk.pose.position.y = cube_pos(1);
//   mk.pose.position.z = cube_pos(2);
//   mk.scale.x = cube_scale(0);
//   mk.scale.y = cube_scale(1);
//   mk.scale.z = cube_scale(2);
//   mk.color.a = 0.3;
//   mk.color.r = 1.0;
//   mk.color.g = 0.0;
//   mk.color.b = 0.0;
//   mk.pose.orientation.w = 1.0;
//   mk.pose.orientation.x = 0.0;
//   mk.pose.orientation.y = 0.0;
//   mk.pose.orientation.z = 0.0;

//   update_range_pub_.publish(mk);
// }

void MapROS::publishESDF() {
  double dist;
  pcl::PointCloud<pcl::PointXYZI> cloud;
  pcl::PointXYZI pt;

  const double min_dist = 0.0;
  const double max_dist = 3.0;

  Eigen::Vector3i min_cut = map_->md_->local_bound_min_ - Eigen::Vector3i(map_->mp_->local_map_margin_,
                                                                          map_->mp_->local_map_margin_,
                                                                          map_->mp_->local_map_margin_);
  Eigen::Vector3i max_cut = map_->md_->local_bound_max_ + Eigen::Vector3i(map_->mp_->local_map_margin_,
                                                                          map_->mp_->local_map_margin_,
                                                                          map_->mp_->local_map_margin_);
  map_->boundIndex(min_cut);
  map_->boundIndex(max_cut);
  esdf_slice_height_ = camera_pos_.z();

  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y) {
      Eigen::Vector3d pos;
      map_->indexToPos(Eigen::Vector3i(x, y, 1), pos);
      pos(2) = esdf_slice_height_;
      dist = map_->getDistance(pos);
      // dist = min(dist, max_dist);
      // dist = max(dist, min_dist);
      pt.x = pos(0);
      pt.y = pos(1);
      pt.z = -0.2;
      // pt.intensity = (dist - min_dist) / (max_dist - min_dist);
      pt.intensity = dist;
      cloud.push_back(pt);
    }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);

  esdf_pub_.publish(cloud_msg);

  // ROS_INFO("pub esdf");
}


}