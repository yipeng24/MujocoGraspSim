#pragma once
#include "util_gym/util_gym.hpp"
#include <plan_env_lod/sdf_map.h>
#include <quadrotor_msgs/SyncFrame.h>
namespace map_interface
{
class MapInterface{
 private:
  std::shared_ptr<airgrasp::SDFMap> tgt_map_ptr_;
  std::shared_ptr<airgrasp::SDFMap> env_map_ptr_;
  ros::NodeHandle nh_;
  ros::Subscriber sync_sub_;
  Eigen::Vector3d cam_pos_last_;
  Eigen::Quaterniond cam_quat_last_;
  ros::Time last_time_;
  bool use_world_pcl_ = false;
  
  public:
    MapInterface(ros::NodeHandle &nh)
    {
        ROS_INFO("[MapInterface] initializing");
        // 1️⃣ 初始化 tgt_map
        tgt_map_ptr_ = std::make_shared<airgrasp::SDFMap>();
        initSingleMap(nh, "tgt_map", tgt_map_ptr_);
        ROS_INFO("[MapInterface] tgt_map initialized");

        // 2️⃣ 初始化 env_map
        env_map_ptr_ = std::make_shared<airgrasp::SDFMap>();
        initSingleMap(nh, "env_map", env_map_ptr_);
        ROS_INFO("[MapInterface] env_map initialized");

        nh.param("use_world_pcl", use_world_pcl_, false);

        // 3️⃣ 订阅 SyncFrame 话题
        std::string syncframe_topic;
        nh.param("syncframe_topic", syncframe_topic, std::string("/sync_frame_pcl"));

        if(use_world_pcl_)
          sync_sub_ = nh.subscribe(syncframe_topic, 10, &MapInterface::syncFrameWorldCallback, this);
        else 
          sync_sub_ = nh.subscribe(syncframe_topic, 10, &MapInterface::syncFrameCallback, this);
    }


  private:

    void initSingleMap(ros::NodeHandle &nh, const std::string &prefix, std::shared_ptr<airgrasp::SDFMap> map){
        std::cout << "Initializing " << prefix << std::endl;  
        map->initMap(nh);
        std::cout << "Setting parameters for " << prefix << std::endl;

        double res;
        Eigen::Vector3d map_size, map_center;
        nh.param(prefix + "/resolution", res, -1.0);
        nh.param(prefix + "/map_size_x", map_size(0), -1.0);
        nh.param(prefix + "/map_size_y", map_size(1), -1.0);
        nh.param(prefix + "/map_size_z", map_size(2), -1.0);
        nh.param(prefix + "/map_center_x", map_center(0), -1.0);
        nh.param(prefix + "/map_center_y", map_center(1), -1.0);
        nh.param(prefix + "/map_center_z", map_center(2), -1.0);

        std::cout << prefix << " res: " << res << std::endl;
        std::cout << prefix << " map_size: " << map_size.transpose() << std::endl;
        std::cout << prefix << " map_center: " << map_center.transpose() << std::endl;
        
        map->setMapParam(res, map_size, map_center, prefix);
    }

    void syncFrameCallback(const quadrotor_msgs::SyncFrame::ConstPtr& msg){

        // 取出 body_odom
        const nav_msgs::Odometry& body_odom = msg->body_odom;
        Eigen::Vector3d position;
        position << msg->body_odom.pose.pose.position.x,
                    msg->body_odom.pose.pose.position.y,
                    msg->body_odom.pose.pose.position.z;

        Eigen::Quaterniond cam_quat;
        cam_quat.x() = msg->body_odom.pose.pose.orientation.x;
        cam_quat.y() = msg->body_odom.pose.pose.orientation.y;
        cam_quat.z() = msg->body_odom.pose.pose.orientation.z;
        cam_quat.w() = msg->body_odom.pose.pose.orientation.w;

        // double err = quatAngularDistanceRad(cam_quat, cam_quat_last_);
        // if(err < 30.0/180.0*M_PI && (position - cam_pos_last_).norm() < 0.05 && (msg->header.stamp - last_time_).toSec() < 0.1){
        //     ROS_WARN_STREAM("ros & pos & timeStamp change too small, skip this frame. err: " << err*180.0/M_PI << " deg");
        //     return;
        // }
        // cam_quat_last_ = cam_quat;
        // cam_pos_last_ = position;
        // last_time_ = msg->header.stamp;

        pcl::PointCloud<pcl::PointXYZ> cloud;
        
        pcl::fromROSMsg(msg->tgt_pcl, cloud);
        tgt_map_ptr_->inputPointCloud(cloud, cloud.points.size(), position);
        tgt_map_ptr_->vis();

        pcl::fromROSMsg(msg->env_pcl, cloud);
        env_map_ptr_->inputPointCloud(cloud, cloud.points.size(), position);
        env_map_ptr_->vis();
    }

    void syncFrameWorldCallback(const quadrotor_msgs::SyncFrame::ConstPtr& msg){
        pcl::PointCloud<pcl::PointXYZ> cloud_tgt, cloud_env;

        // tgt_map 不再用于导航/碰撞优化，跳过 ESDF 重算避免 CPU 尖峰导致物理循环抖动
        pcl::fromROSMsg(msg->tgt_pcl, cloud_tgt);

        pcl::fromROSMsg(msg->env_pcl, cloud_env);
        env_map_ptr_->inputPointCloudWorld(cloud_env);
        env_map_ptr_->setPointCloudFree(cloud_tgt);
        env_map_ptr_->vis();
    }

    double quatAngularDistanceRad(const Eigen::Quaterniond& q1,
                                        const Eigen::Quaterniond& q2)
    {
        // 归一化，避免非单位带来的误差
        Eigen::Quaterniond a = q1.normalized();
        Eigen::Quaterniond b = q2.normalized();

        // 四元数点积（注意 q 与 -q 等价，取绝对值得到最小角）
        double d = std::abs(a.coeffs().dot(b.coeffs()));   // coeffs() 顺序为 (x,y,z,w)

        // 数值钳制，防止 acos(>1 or < -1)
        d = std::min(1.0, std::max(-1.0, d));

        // 角度（弧度）：2 * arccos(dot)
        return 2.0 * std::acos(d);
    }

  public:
    // ------ partical ------
    inline const bool isOccupied(const Eigen::Vector3d& p) const {
      return !env_map_ptr_->isValid(p);
    }

    inline const bool isOccupied(const Eigen::Vector3i& id) const {
      return !env_map_ptr_->isValid(id);
    }

    inline bool checkRayValid(const Eigen::Vector3d& p0, const Eigen::Vector3d& p1) const {
      return env_map_ptr_->checkRayValid(p0, p1);
    }
    // ------ partical ------


    // ------ whole-body ------
    // inline const bool isOccupied(const Eigen::VectorXd& state) const {
    //     return env_map_ptr_->isOccupied(state);
    // }

    inline bool checkRayValid(const Eigen::VectorXd& s0, const Eigen::VectorXd& s1) const {
      return env_map_ptr_->checkRayValid(s0,  s1);
    }
    // ------ whole-body ------

    inline double getCostWithGrad(const Eigen::Vector3d& pos, Eigen::Vector3d& grad) const{
      return env_map_ptr_->getDistWithGrad(pos, grad);
    }
    inline const Eigen::Vector3i pos2idx(const Eigen::Vector3d& pt) const {
      Eigen::Vector3i id;
      env_map_ptr_->posToIndex(pt,id);
      return id;
    }
    inline const Eigen::Vector3d idx2pos(const Eigen::Vector3i& id) const {
      Eigen::Vector3d pt;
      env_map_ptr_->indexToPos(id,pt);
      return pt;
    }
    inline double resolution() const{
      return env_map_ptr_->getResolution();
    }

    inline void getAABBPoints(Eigen::MatrixXd& aabb_pts, 
                              const Eigen::Vector3d& start_pos, 
                              const Eigen::Vector3d& end_pos,
                              const Eigen::Vector3d& box_size) const{
      env_map_ptr_->getAABBPoints(aabb_pts, start_pos, end_pos, box_size);
    }

    inline void getAABBPoints(std::vector<Eigen::Vector3d>& aabb_pts,
                              const Eigen::Vector3d& start_pos,
                              const Eigen::Vector3d& end_pos,
                              const Eigen::Vector3d& box_size) const{
        // 只用环境地图做碰撞优化；tgt_map 是抓取目标，不作为导航障碍
        std::vector<Eigen::Vector3d> aabb_pts_env;
        env_map_ptr_->getAABBPoints(aabb_pts_env, start_pos, end_pos, box_size);
        aabb_pts_env = env_map_ptr_->farthestPointSampling(aabb_pts_env, 2000);

        aabb_pts = aabb_pts_env;

        if (aabb_pts.empty()) {
            std::cout << "Warning: getAABBPoints returned 0 points!" << std::endl;
            return;
        }

        std::cout << "getAABBPoints get " << aabb_pts.size() << " pts" << std::endl;
        std::cout << "aabb_pts: " << aabb_pts[0].transpose() << std::endl;
    }

    inline void getAABBPointsSample(Eigen::MatrixXd& aabb_pts, 
                              const std::vector<Eigen::Vector3d>& sample_pts,
                              const Eigen::Vector3d& box_size) const{
      // env_map_ptr_->getAABBPoints(aabb_pts, sample_pts);
    }

    inline void getAABBPointsSample(std::vector<Eigen::Vector3d>& aabb_pts,
                                    const std::vector<Eigen::Vector3d>& sample_pts,
                                    const Eigen::Vector3d& box_size) const {
        // 只用环境地图；tgt_map 是抓取目标，不作为导航障碍
        env_map_ptr_->getAABBPointsSample(aabb_pts, sample_pts, box_size);
    }


    inline void getAABBPointsSample(std::vector<Eigen::Vector3d>& aabb_pts_env,
                                    std::vector<Eigen::Vector3d>& aabb_pts_tgt,
                                    const std::vector<Eigen::Vector3d>& sample_pts,
                                    const Eigen::Vector3d& box_size) const{
      env_map_ptr_->getAABBPointsSample(aabb_pts_env, sample_pts, box_size);
      tgt_map_ptr_->getAABBPointsSample(aabb_pts_tgt, sample_pts, box_size);
    }


    inline Eigen::Vector3d getMinBound() const{
      return env_map_ptr_->getMinBound();
    }

    inline Eigen::Vector3d getMaxBound() const{
      return env_map_ptr_->getMaxBound();
    }

};


} // namespace map_interface