#include "ch_rc_sdf/ch_rc_sdf.h"
#include <memory>

namespace clutter_hand {

CH_RC_SDF::CH_RC_SDF() = default;

CH_RC_SDF::~CH_RC_SDF() = default;

void CH_RC_SDF::initMap(ros::NodeHandle &nh, bool init_field, bool vis_en) {

  nh_ = nh;
  vis_en_ = vis_en;

  //! 4. Init visualization
  vis_ptr_ = std::make_shared<visualization_rc_sdf::Visualization>(nh);

  //! get collision map check size
  nh.param("col_check_map_size/x", col_check_map_size_.x(), 2.0);
  nh.param("col_check_map_size/y", col_check_map_size_.y(), 2.0);
  nh.param("col_check_map_size/z", col_check_map_size_.z(), 2.0);
  std::cout << "[CH_RC_SDF]: col_check_map_size: "
            << col_check_map_size_.transpose() << std::endl;

  //! 1. Get boxed params
  nh.param("ch_rc_sdf/box_num", box_num_, 1);
  nh.param("ch_rc_sdf/resolution", resolution_, 0.1);
  nh.param("ch_rc_sdf/default_dist", default_dist_, 5.0);
  resolution_inv_ = 1.0 / resolution_;
  std::cout << "[CH_RC_SDF]: box_num: " << box_num_ << std::endl;
  std::cout << "[CH_RC_SDF]: resolution: " << resolution_ << std::endl;
  std::cout << "[CH_RC_SDF]: default_dist: " << default_dist_ << std::endl;
  thetas_est_.resize(box_num_ - 1);
  dthetas_est_.resize(box_num_ - 1);

  //! 2. Initialize SDF Map
  sdf_map_ptr_ = std::make_shared<SDFMap>();
  sdf_map_ptr_->loadAndInitialize(nh, box_num_, resolution_, default_dist_);
  
  // Copy SDF box data to box_data_list_ for backward compatibility with visualization
  box_data_list_.clear();
  for (int i = 0; i < box_num_; i++) {
    FieldData field_data;
    const SDFBoxData& sdf_box = sdf_map_ptr_->getBoxData(i);
    
    // Copy field parameters
    field_data.field_size = sdf_box.field_size;
    field_data.field_min_boundary = sdf_box.field_min_boundary;
    field_data.field_max_boundary = sdf_box.field_max_boundary;
    field_data.field_origin = sdf_box.field_origin;
    field_data.field_voxel_num = sdf_box.field_voxel_num;
    field_data.distance_buffer = sdf_box.distance_buffer;
    field_data.shell_thickness = sdf_box.shell_thickness;
    field_data.disk_params_list = sdf_box.disk_params_list;
    field_data.line_params_list = sdf_box.line_params_list;
    
    // Note: T_cur2last and ang_id are no longer loaded here
    // They are managed by kine_ptr_ instead
    field_data.T_cur2last.setZero();
    field_data.ang_id = 0;
    
    box_data_list_.push_back(field_data);
    
    // Load initial angle for theta estimation
    double init_ang;
    nh.param("ch_rc_sdf/box" + std::to_string(i) + "/Ang_2_last/init_ang", init_ang, 0.0);
    if (i > 0) {
      thetas_est_[i - 1] = init_ang;
      dthetas_est_[i - 1] = 0.0;
      std::cout << "[CH_RC_SDF]: box_" << i << " init_ang: " << thetas_est_[i - 1] << std::endl;
    }
  }

  //! Initialize Kinematics Module
  kine_ptr_ = std::make_shared<RobotKinematics>();
  std::vector<KinematicLink> links_temp = RobotKinematics::loadKinematicLinksFromROS(nh, box_num_);
  kine_ptr_->setKinematicParams(links_temp);
  std::cout << "[CH_RC_SDF]: Kinematics module initialized." << std::endl;

  // ! subscriber init
  sub_slice_coord_ =
      nh.subscribe("slice_coord", 10, &CH_RC_SDF::callback_slice_coord, this);
  sub_odom = nh.subscribe("odom", 100, &CH_RC_SDF::odom_callback, this);
  sub_joint_state_est = nh.subscribe("joint_state_est", 100,
                                     &CH_RC_SDF::joint_state_est_cb, this);

  if (vis_en_) {
    last_vis_time_ = ros::Time::now();

    sub_uam_state_vis_ = nh.subscribe("uam_state_goal_vis", 30,
                                      &CH_RC_SDF::uam_state_vis_callback, this);

    // vis publisher init
    visualization_msgs::MarkerArray marker_array;
    robotMarkersPub(marker_array, "robot");
    robotMarkersPub(marker_array, "debug_robot_markers");
    robotMarkersPub(marker_array, "rrt_whole_body_path");

    // 初始化 Service
    body_pose_srv_ = nh.advertiseService("get_base_pose", &CH_RC_SDF::getBasePoseCallback, this);
  }
}

void CH_RC_SDF::getCurEndPose(const Eigen::Vector3d &drone_pos,
                              const Eigen::Quaterniond &drone_q,
                              Eigen::Vector3d &end_pos,
                              Eigen::Quaterniond &end_q) {
  kine_ptr_->getCurEndPose(drone_pos, drone_q, thetas_est_, end_pos, end_q);
}

// ===========================================
// SDF Functions
// ===========================================

// Wrapper function for backward compatibility
double CH_RC_SDF::getRoughDistInFrameBox(const Eigen::Vector3d &pt, const int &box_id) {
  return sdf_map_ptr_->getRoughDistInFrameBox(pt, box_id);
}

double CH_RC_SDF::getRoughDist_body(const Eigen::VectorXd &arm_angles,
                                    const Eigen::Vector3d &pt, int &box_id) {
  double dist = default_dist_;

  for (int id = 0; id < box_data_list_.size(); ++id) {
    double dist_temp;
    Eigen::Vector3d pt_box;
    kine_ptr_->convertPosToFrame(arm_angles, pt, 0, id, pt_box);

    dist_temp = sdf_map_ptr_->getRoughDistInFrameBox(pt_box, id);
    if (dist_temp < dist) {
      dist = dist_temp;
      box_id = id;
    }
  }
  return dist;
}

// 查找p_body在box_id中sdf值和梯度
double CH_RC_SDF::getDistWithGradInBox(const Eigen::VectorXd &arm_angles,
                                       const Eigen::Vector3d &p_body,
                                       const int &box_id,
                                       Eigen::Vector3d &grad_body,
                                       Eigen::Vector3d &grad_box) {
  Eigen::Vector3d p_box;
  kine_ptr_->convertPosToFrame(arm_angles, p_body, 0, box_id, p_box);

  if (!isInBox(p_box, box_id)) {
    grad_body.setZero();
    grad_box.setZero();
    return default_dist_;
  }

  double dist = sdf_map_ptr_->getDistWithGradInFrameBox(p_box, box_id, grad_box);
  kine_ptr_->convertPosToFrame(arm_angles, grad_box, box_id, 0, grad_body);

  return dist;
}

// 找到sdf最小的box
double CH_RC_SDF::getDistWithGrad_body(const Eigen::VectorXd &arm_angles,
                                       const Eigen::Vector3d &pt, int &box_id,
                                       Eigen::Vector3d &grad) {
  double dist = default_dist_;
  grad.setZero();

  if (isInMap(pt) == false)
    return dist;

  for (int id = 0; id < box_data_list_.size(); ++id) {
    double dist_temp;
    Eigen::Vector3d grad_temp;
    Eigen::Vector3d pt_box;
    kine_ptr_->convertPosToFrame(arm_angles, pt, 0, id, pt_box);

    dist_temp = sdf_map_ptr_->getDistWithGradInFrameBox(pt_box, id, grad_temp);
    if (dist_temp < dist) {
      dist = dist_temp;
      grad = grad_temp;
      box_id = id;
    }
  }
  return dist;
}

Eigen::VectorXd
CH_RC_SDF::get_grad_thetas_sdf(const Eigen::VectorXd &thetas,
                               const Eigen::Vector3d &p_body,
                               const Eigen::Vector3d &gd_box, const int &box_id,
                               const std::vector<Eigen::Matrix4d> &transforms) {
  Eigen::VectorXd grad_thetas(thetas.size());
  grad_thetas.setZero();

  Eigen::Vector4d p_body_homo = Eigen::Vector4d::Ones();
  p_body_homo.head(3) = p_body;

  //! 计算 p_box_id 对 thetas(i) 的偏导
  for (size_t i = 0; i < box_id; ++i) {
    Eigen::Matrix4d transform_left, transform_right, transform_gd;
    transform_left.setIdentity();
    transform_right.setIdentity();
    transform_gd.setZero();

    // 具体计算
    for (size_t j = 0; j < box_id; ++j) {
      if (j < i)
        transform_right = transforms[j] * transform_right;

      if (j > i)
        transform_left = transforms[j] * transform_left;

      // 计算 d T(i) / d theta(i)
      if (j == i) {
        double theta = thetas(j);
        switch (box_data_list_[j + 1].ang_id) {
        case 0:
          transform_gd << 0, 0, 0, 0, 0, -sin(theta), -cos(theta), 0, 0,
              cos(theta), -sin(theta), 0, 0, 0, 0, 0;
          break;
        case 1:
          transform_gd << -sin(theta), 0, cos(theta), 0, 0, 0, 0, 0,
              -cos(theta), 0, -sin(theta), 0, 0, 0, 0, 0;
          break;
        case 2:
          transform_gd << -sin(theta), -cos(theta), 0, 0, cos(theta),
              -sin(theta), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
          break;
        }
      }
    }
    Eigen::Vector3d dp_box_id_dtheta =
        (transform_left * transform_gd * transform_right * p_body_homo).head(3);
    grad_thetas(i) = gd_box.dot(dp_box_id_dtheta);
  }
  return grad_thetas;
}

Eigen::VectorXd CH_RC_SDF::get_grad_thetas_wp(
    const Eigen::Vector3d &pos, const Eigen::Quaterniond &q,
    const Eigen::VectorXd &thetas, const Eigen::Matrix4d &T_e2b,
    const Eigen::Vector3d &d_dis_d_p_e,
    const std::vector<Eigen::Matrix4d> &transforms) {
  Eigen::VectorXd grad_thetas(thetas.size());
  grad_thetas.setZero();

  Eigen::Matrix4d T_b2w = Eigen::Matrix4d::Identity();
  T_b2w.block<3, 3>(0, 0) = q.toRotationMatrix();
  T_b2w.block<3, 1>(0, 3) = pos;

  Eigen::Vector4d p_end_homo;
  p_end_homo << 0, 0, 0, 1;
  //! 计算 p_end 对 thetas(i) 的偏导
  for (size_t k = 0; k < getBoxNum() - 1; ++k) {
    Eigen::Matrix4d transform_left, transform_right, transform_gd;
    transform_left = T_b2w;
    transform_right.setIdentity();
    transform_gd.setZero();

    // 具体计算
    for (size_t l = 0; l < getBoxNum() - 1; ++l) {
      if (l < k)
        transform_left = transform_left * transforms[l];

      if (l > k)
        transform_right = transform_right * transforms[l];

      // 计算 d T(i) / d theta(i)
      if (l == k) {
        double theta = thetas(l);
        switch (kine_ptr_->getLinks()[l + 1].ang_id) {
        case 0:
          transform_gd << 0, 0, 0, 0, 0, -sin(theta), -cos(theta), 0, 0,
              cos(theta), -sin(theta), 0, 0, 0, 0, 0;
          break;
        case 1:
          transform_gd << -sin(theta), 0, cos(theta), 0, 0, 0, 0, 0,
              -cos(theta), 0, -sin(theta), 0, 0, 0, 0, 0;
          break;
        case 2:
          transform_gd << -sin(theta), -cos(theta), 0, 0, cos(theta),
              -sin(theta), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
          break;
        }
        // transform_gd = -transforms[l] * transform_gd * transforms[l];
        transform_gd = -transform_left * transform_gd * transform_right;
      }
    }
    Eigen::Vector3d dp_end_dtheta =
        (transform_left * transform_gd * transform_right * p_end_homo).head(3);
    grad_thetas(k) = d_dis_d_p_e.dot(dp_end_dtheta);
  }
  return grad_thetas;
}

bool CH_RC_SDF::isInRob(const Eigen::VectorXd &arm_angles,
                        const Eigen::Vector3d &pt_body) {
  double dist_threshold = 0.2;
  for (int id = 0; id < box_data_list_.size(); ++id) {
    Eigen::Vector3d pt_box;
    kine_ptr_->convertPosToFrame(arm_angles, pt_body, 0, id, pt_box);
    if (getRoughDistInFrameBox(pt_box, id) < dist_threshold)
      return true;
  }
  return false;
}

void CH_RC_SDF::get_grad_dis_wp_full(const Eigen::Vector3d &pos,
                                     const Eigen::Quaterniond &q,
                                     const Eigen::VectorXd &thetas,
                                     const Eigen::Vector3d &d_dis_d_p_e,
                                     Eigen::Vector3d &d_p, Eigen::Vector4d &d_q,
                                     Eigen::VectorXd &d_theta) {
  // 1. 调用之前定义的函数，获取 3 x (3 + 4 + N) 的雅可比矩阵
  // 列分布: [0-2]: drone_pos, [3-6]: drone_q(w,x,y,z), [7+]: thetas
  Eigen::MatrixXd J_d_p_e = kine_ptr_->getEndPosJacobianFull(pos, q, thetas);

  // 2. 利用链式法则计算全梯度向量 (Size: 7 + N)
  // d_dis_d_full = J^T * (d_dis / d_p_e)
  Eigen::VectorXd d_dis_d_full = J_d_p_e.transpose() * d_dis_d_p_e;

  // 3. 将计算好的全梯度拆分并赋值给输出参数

  // 无人机位置梯度 (3维)
  d_p = d_dis_d_full.segment<3>(0);

  // 无人机四元数梯度 (4维: w, x, y, z)
  d_q = d_dis_d_full.segment<4>(3);

  // 机械臂关节角梯度 (N维)
  d_theta = d_dis_d_full.tail(thetas.size());
}

//! subscriber and publisher
void CH_RC_SDF::joint_state_est_cb(
    const sensor_msgs::JointState::ConstPtr &msg) {
  if (msg->position.size() < box_num_ - 1)
    ROS_ERROR("thetas size is smaller than box_num_-1");

  for (size_t i = 0; i < box_num_ - 1; ++i) {
    thetas_est_[i] = msg->position[i];
    dthetas_est_[i] = msg->velocity[i];
  }
  have_arm_angles_cur_ = true;
}

bool CH_RC_SDF::getBasePoseCallback(quadrotor_msgs::GetBasePose::Request &req,
                                    quadrotor_msgs::GetBasePose::Response &res) {
    // 1. 检查输入 theta 长度
    int n = box_num_ - 1;
    if (req.thetas.size() != n) {
        ROS_ERROR("[CH_RC_SDF]: Input thetas size mismatch! Expected %d, got %ld", n, req.thetas.size());
        res.success = false;
        return true; 
    }

    // 2. 将输入的 thetas 转换为 Eigen 向量
    Eigen::VectorXd arm_angles = Eigen::Map<const Eigen::VectorXd>(req.thetas.data(), n);

    // 3. 构建末端在世界系下的变换矩阵 T_e2w
    Eigen::Quaterniond q_e2w(req.end_pose.orientation.w, 
                             req.end_pose.orientation.x, 
                             req.end_pose.orientation.y, 
                             req.end_pose.orientation.z);
    Eigen::Matrix4d T_e2w = Eigen::Matrix4d::Identity();
    T_e2w.block<3, 3>(0, 0) = q_e2w.normalized().toRotationMatrix();
    T_e2w.block<3, 1>(0, 3) << req.end_pose.position.x, 
                               req.end_pose.position.y, 
                               req.end_pose.position.z;

    // 4. 计算末端相对于机体的变换矩阵 T_e2b
    // 根据 robot_kinematics.hpp, box_num_-1 是末端, 0 是机体(基座)
    Eigen::Matrix4d T_e2b;
    kine_ptr_->getRelativeTransform(arm_angles, box_num_ - 1, 0, T_e2b);

    // 5. 计算机体在世界系下的位姿 T_b2w = T_e2w * inv(T_e2b)
    // 注意：T_e2b 本身就是从 frame e 到 frame b 的变换
    // 所以 T_b2w = T_e2w * T_b2e (即 T_e2w * T_e2b.inverse())
    Eigen::Matrix4d T_b2w = T_e2w * kine_ptr_->inverseTransform(T_e2b);

    // 6. 填充响应结果
    Eigen::Vector3d p_b2w = T_b2w.block<3, 1>(0, 3);
    Eigen::Quaterniond q_b2w(T_b2w.block<3, 3>(0, 0));

    res.body_pose.position.x = p_b2w.x();
    res.body_pose.position.y = p_b2w.y();
    res.body_pose.position.z = p_b2w.z();
    res.body_pose.orientation.w = q_b2w.w();
    res.body_pose.orientation.x = q_b2w.x();
    res.body_pose.orientation.y = q_b2w.y();
    res.body_pose.orientation.z = q_b2w.z();
    
    res.success = true;
    return true;
}

} // namespace clutter_hand