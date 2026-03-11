#include <traj_opt/lbfgs_raw.hpp>
#include <traj_opt/traj_opt.h>

namespace traj_opt {

static Eigen::Vector3d g_(0, 0, -9.8);

static Eigen::Vector3d f_N(const Eigen::Vector3d &x) { return x.normalized(); }
static Eigen::MatrixXd f_DN(const Eigen::Vector3d &x) {
  double x_norm_2 = x.squaredNorm();
  return (Eigen::MatrixXd::Identity(3, 3) - x * x.transpose() / x_norm_2) /
         sqrt(x_norm_2);
}
static Eigen::MatrixXd f_D2N(const Eigen::Vector3d &x,
                             const Eigen::Vector3d &y) {
  double x_norm_2 = x.squaredNorm();
  double x_norm_3 = x_norm_2 * x.norm();
  Eigen::MatrixXd A =
      (3 * x * x.transpose() / x_norm_2 - Eigen::MatrixXd::Identity(3, 3));
  return (A * y * x.transpose() - x * y.transpose() -
          x.dot(y) * Eigen::MatrixXd::Identity(3, 3)) /
         x_norm_3;
}

static Eigen::Matrix3d getQuatTransDW(
    const Eigen::Vector4d &quat) { // 旋转矩阵R转成四元数quat，返回R.transpose()
                                   // 的四元数系数W的求导
  Eigen::Matrix3d ret;
  double w = quat(0);
  double x = quat(1);
  double y = quat(2);
  double z = quat(3);
  ret << 0, 2 * z, -2 * y, -2 * z, 0, 2 * x, 2 * y, -2 * x, 0;
  return ret;
}

static Eigen::Matrix3d getQuatTransDX(
    const Eigen::Vector4d &quat) { // 旋转矩阵R转成四元数quat，返回R.transpose()
                                   // 的四元数系数W的求导
  Eigen::Matrix3d ret;
  double w = quat(0);
  double x = quat(1);
  double y = quat(2);
  double z = quat(3);
  ret << 0, 2 * y, 2 * z, 2 * y, -4 * x, 2 * w, 2 * z, -2 * w, -4 * x;
  return ret;
}

static Eigen::Matrix3d getQuatTransDY(
    const Eigen::Vector4d &quat) { // 旋转矩阵R转成四元数quat，返回R.transpose()
                                   // 的四元数系数W的求导
  Eigen::Matrix3d ret;
  double w = quat(0);
  double x = quat(1);
  double y = quat(2);
  double z = quat(3);
  ret << -4 * y, 2 * x, -2 * w, 2 * x, 0, 2 * z, 2 * w, 2 * z, -4 * y;
  return ret;
}

static Eigen::Matrix3d getQuatTransDZ(
    const Eigen::Vector4d &quat) { // 旋转矩阵R转成四元数quat，返回R.transpose()
                                   // 的四元数系数W的求导
  Eigen::Matrix3d ret;
  double w = quat(0);
  double x = quat(1);
  double y = quat(2);
  double z = quat(3);
  ret << -4 * z, 2 * w, 2 * x, -2 * w, -4 * z, 2 * y, 2 * x, 2 * y, 0;
  return ret;
}

static Eigen::Matrix3d gradRot2Angle(const double &angle, const int &axis) {
  Eigen::Matrix3d ret;

  switch (axis) {
  case 0:
    ret << 0, 0, 0, 0, -sin(angle), -cos(angle), 0, cos(angle), -sin(angle);

  case 1:
    ret << -sin(angle), 0, cos(angle), 0, 0, 0, -cos(angle), 0, -sin(angle);

  case 2:
    ret << -sin(angle), -cos(angle), 0, cos(angle), -sin(angle), 0, 0, 0, 0;
  }
  return ret;
}

bool TrajOpt::grad_sdf_full_state(
    const Eigen::Vector3d &pt_obs, const Eigen::Vector3d &pos_ego,
    const Eigen::Vector4d &quat, const Eigen::VectorXd &thetas,
    Eigen::Vector3d &gradp, Eigen::Vector4d &grad_quat, Eigen::VectorXd &grad_thetas, double &cost_sdf) {

  // 1. 初始化
  gradp.setZero();
  grad_quat.setZero();
  grad_thetas.resize(thetas.size());
  grad_thetas.setZero();
  cost_sdf = 0.0;

  Eigen::Quaterniond q_b2w(quat(0), quat(1), quat(2), quat(3));
  Eigen::Matrix3d R_b2w = q_b2w.toRotationMatrix();

  // 障碍物相对于无人机的位置向量 (World系)
  Eigen::Vector3d r_w = pt_obs - pos_ego;
  // 障碍物在机体系下的位置
  Eigen::Vector3d p_body = R_b2w.transpose() * r_w;

  if (!rc_sdf_ptr_->isInMap(p_body))
    return false;

  int box_num = rc_sdf_ptr_->getBoxNum();
  
  for (int i = 0; i < box_num; i++) {
    if (i == 2) continue; // 跳过特定 Box

    Eigen::Vector3d grad_sdf_body, grad_sdf_box;
    // 获取 SDF 值及在机体系下的梯度方向 grad_sdf_body
    double sdf = rc_sdf_ptr_->getDistWithGradInBox(thetas, p_body, i,
                                                   grad_sdf_body, grad_sdf_box);

    if (sdf > dist_threshold_) continue;

    double cost_temp, grad_out;
    double mu = 0.01;
    if (smoothedL1(dist_threshold_ - sdf, mu, cost_temp, grad_out)) {
      
      // A. 位置梯度: d(cost)/dp_ego = grad_out * R_b2w * grad_sdf_body
      // 解释：无人机移向障碍物，距离减小，代价增加
      gradp += grad_out * (R_b2w * grad_sdf_body);

      // B. 四元数梯度: 使用 getQuatTrans 系列函数 (对应 R^T 的导数)
      // d(cost)/dq = -grad_out * d(dist)/dq
      // d(dist)/dq = g_body^T * d(R^T)/dq * r_w
      Eigen::Vector4d gd_q;
      gd_q(0) = grad_sdf_body.transpose() * getQuatTransDW(quat) * r_w;
      gd_q(1) = grad_sdf_body.transpose() * getQuatTransDX(quat) * r_w;
      gd_q(2) = grad_sdf_body.transpose() * getQuatTransDY(quat) * r_w;
      gd_q(3) = grad_sdf_body.transpose() * getQuatTransDZ(quat) * r_w;
      
      grad_quat -= grad_out * gd_q;

      // C. 关节角梯度: 使用几何雅可比 (针对 p_body 点)
      for (int j = 0; j < i; ++j) {
        if (j >= thetas.size()) break;

        Eigen::Matrix4d T_j_0; 
        rc_sdf_ptr_->kine_ptr_->getRelativeTransform(thetas, j, 0, T_j_0);
        
        // 获取第 j+1 个 link 的旋转轴 (对应 control thetas[j] 的关节)
        int ang_id = rc_sdf_ptr_->kine_ptr_->getLinks()[j + 1].ang_id;
        Eigen::Vector3d axis_local = Eigen::Vector3d::Zero();
        axis_local(ang_id) = 1.0;
        Eigen::Vector3d z_j_b = T_j_0.block<3, 3>(0, 0) * axis_local;
        Eigen::Vector3d p_j_b = T_j_0.block<3, 1>(0, 3);

        // 几何雅可比: 关节转动导致障碍物在 Box 系下投影的变化
        // d(dist)/d_theta = z · (grad_sdf_body x (p_body - p_j_b))
        double d_dist_d_theta = z_j_b.dot(grad_sdf_body.cross(p_body - p_j_b));
        
        grad_thetas(j) -= grad_out * d_dist_d_theta;
      }

      cost_sdf += cost_temp;
    }
  }

  gradp *= rhoP_;
  grad_quat *= rhoP_;
  grad_thetas *= rhoP_;
  cost_sdf *= rhoP_;

  return (cost_sdf > 1e-9);
}

// /**
//  * @description:
//  * @param pts_obs  positions of obstacles
//  * @param pos_ego  position of drone
//  * @param quat     orientation of drone
//  * @param arm_angles  arm angle of drone
//  * @return {*}
//  */
bool TrajOpt::grad_sdf_full_state_jr(
    const Eigen::Vector3d &pt_obs, const Eigen::Vector3d &pos_ego,
    const Eigen::Vector4d &quat, const Eigen::VectorXd &thetas,
    const double &yaw, Eigen::Vector3d &gradp,
    Eigen::Vector4d &grad_quat, // 0 -> w, 1 -> x, 2 -> y, 3 -> z
    double &grad_yaw, Eigen::VectorXd &grad_thetas, double &cost_sdf) {
  gradp.setZero();
  grad_quat.setZero();
  grad_yaw = 0;
  grad_thetas.resize(thetas.size());
  grad_thetas.setZero();
  cost_sdf = 0;

  Eigen::Vector4d gd_quat_temp;
  double gd_temp_yaw, cost_sdf_temp;

  Eigen::Quaterniond q_b2w;
  q_b2w.w() = quat(0);
  q_b2w.x() = quat(1);
  q_b2w.y() = quat(2);
  q_b2w.z() = quat(3);

  Eigen::Vector3d pt_body, p_minus_x;
  Eigen::Vector3d grad_sdf_body, grad_sdf_world, grad_sdf_box;
  p_minus_x = pt_obs - pos_ego;
  pt_body = q_b2w.inverse() * p_minus_x;

  std::vector<Eigen::Matrix4d> transforms;
  for (size_t i = 0; i < rc_sdf_ptr_->getBoxNum() - 1; ++i) {
    Eigen::Matrix4d transform;
    rc_sdf_ptr_->kine_ptr_->getRelativeTransform(thetas, i, i + 1, transform);
    transforms.push_back(transform);
  }

  if (!rc_sdf_ptr_->isInMap(pt_body))
    return false;

  for (int i = 0; i < rc_sdf_ptr_->getBoxNum(); i++) {
    if (i == 2)
      continue;
    double sdf = rc_sdf_ptr_->getDistWithGradInBox(thetas, pt_body, i,
                                                   grad_sdf_body, grad_sdf_box);
    if (sdf > dist_threshold_)
      continue;

    grad_sdf_world = q_b2w * grad_sdf_body;

    double grad_out = 0.0;
    double mu = 0.01;
    if (smoothedL1(dist_threshold_ - sdf, mu, cost_sdf_temp, grad_out)) {
      gradp += grad_out * grad_sdf_world;

      gd_quat_temp(0) =
          grad_sdf_body.transpose() * getQuatTransDW(quat) * p_minus_x;
      gd_quat_temp(1) =
          grad_sdf_body.transpose() * getQuatTransDX(quat) * p_minus_x;
      gd_quat_temp(2) =
          grad_sdf_body.transpose() * getQuatTransDY(quat) * p_minus_x;
      gd_quat_temp(3) =
          grad_sdf_body.transpose() * getQuatTransDZ(quat) * p_minus_x;
      grad_quat += -grad_out * gd_quat_temp;

      // Eigen::Matrix3d VR_theta = Eigen::Matrix3d::Zero();
      // VR_theta(0,0) = -sin(yaw);
      // VR_theta(0,1) = -cos(yaw);
      // VR_theta(1,0) = cos(yaw);
      // VR_theta(1,1) = -sin(yaw);
      // VR_theta(2,2) = 1;
      // grad_yaw += -grad_out * grad_sdf_body.transpose() *
      // (VR_theta.transpose() * p_minus_x);

      // std::cout << "grad_quat: " << grad_quat.transpose() << std::endl;

      if (i > 0) {
        Eigen::VectorXd grad_thetas_temp;
        grad_thetas_temp = rc_sdf_ptr_->get_grad_thetas_sdf(
            thetas, pt_body, grad_sdf_box, i, transforms);
        grad_thetas += grad_out * grad_thetas_temp;
      }

      cost_sdf += cost_sdf_temp;

      // INFO_MSG("cost_sdf: " << cost_sdf);
    }
  }

  gradp *= rhoP_;
  grad_quat *= rhoP_;
  grad_yaw *= rhoP_;
  grad_thetas *= rhoP_;
  cost_sdf *= rhoP_;

  if (cost_sdf > DBL_EPSILON)
    return true;

  return false;
}

bool TrajOpt::grad_end_pose(
    const Eigen::Vector3d &p_end_goal, const Eigen::Quaterniond &q_end_goal,
    const Eigen::Vector3d &p_body_goal, const Eigen::Vector3d &pos_ego,
    const Eigen::Vector4d &quat, const Eigen::VectorXd &thetas,
    const double &yaw, Eigen::Vector3d &p_end, Eigen::Vector3d &gradp,
    Eigen::Vector4d &grad_quat, Eigen::VectorXd &grad_thetas, double &cost) {

  // ! 1. 初始化与正运动学
  gradp.setZero();
  grad_quat.setZero();
  grad_thetas.resize(thetas.size());
  grad_thetas.setZero();
  cost = 0;

  Eigen::Quaterniond q_b2w(quat(0), quat(1), quat(2), quat(3));
  Eigen::Vector3d p_e;
  Eigen::Quaterniond q_e2w;
  Eigen::Matrix4d T_e2b;
  rc_sdf_ptr_->kine_ptr_->getEndPose(pos_ego, q_b2w, thetas, p_e, q_e2w, T_e2b);
  p_end = p_e;

  // ========== 配置参数 ==========
  double mu = 0.01;
  double dist_threshold = 0.01;      // 位置允许误差 (m)
  double angle_threshold_deg = 5.0;  // 允许的角度误差 (度)
  
  // 将角度误差转换为旋转矩阵距离阈值
  // dis_rot = (3 - tr(Re * Rg^T)) / 2，对应 1 - cos(theta)
  double angle_rad = angle_threshold_deg * M_PI / 180.0;
  double rot_threshold = 1.0 - std::cos(angle_rad);
  // =============================

  // ! 2. 位置惩罚 (保持原样)
  double dis_pos = (p_e - p_end_goal).norm();
  double cost_pos = 0, grad_pos_mu = 0;
  if (smoothedL1(dis_pos - dist_threshold, mu, cost_pos, grad_pos_mu)) {
      Eigen::Vector3d d_dis_d_pe = (dis_pos > 1e-6) ? (p_e - p_end_goal).normalized() : Eigen::Vector3d::Zero();
      Eigen::Vector3d gp; Eigen::Vector4d gq; Eigen::VectorXd gt;
      rc_sdf_ptr_->get_grad_dis_wp_full(pos_ego, q_b2w, thetas, d_dis_d_pe, gp, gq, gt);
      
      double weight = grad_pos_mu * rhoWP_;
      gradp += weight * gp;
      grad_quat += weight * gq;
      grad_thetas += weight * gt;
      cost += cost_pos * rhoWP_;
  }

  // ! 3. 姿态惩罚 (旋转矩阵版本)
  Eigen::Matrix3d R_e = q_e2w.toRotationMatrix();
  Eigen::Matrix3d R_goal = q_end_goal.toRotationMatrix();
  Eigen::Matrix3d R_drone = q_b2w.toRotationMatrix();
  Eigen::Matrix3d R_e2b = T_e2b.block<3,3>(0,0);

  // 姿态距离: (3 - Trace(Re * Rg^T)) / 2 
  // 范围 0 到 2 (0度到180度)
  double tr_err = (R_e * R_goal.transpose()).trace();
  double dis_rot = (3.0 - tr_err) / 2.0; 

  double cost_rot = 0, grad_rot_mu = 0;
  double rhoRot = rhoRotFactor_ * rhoWP_;

  if (smoothedL1(dis_rot - rot_threshold, mu, cost_rot, grad_rot_mu)) {
      double weight_rot = grad_rot_mu * rhoRot;

      // 3.1 误差对末端旋转矩阵的偏导: d_dis_rot / d_Re = -0.5 * R_goal
      Eigen::Matrix3d d_dis_d_Re = -0.5 * R_goal;

      // A. 对 Drone Quat 的梯度
      // 需要通过旋转矩阵对四元数的导数(R_mat)进行转换
      for (int j = 0; j < 4; ++j) {
          Eigen::Matrix3d dRdQ_j;
          if (j == 0) dRdQ_j = getQuatTransDW(quat).transpose();
          else if (j == 1) dRdQ_j = getQuatTransDX(quat).transpose();
          else if (j == 2) dRdQ_j = getQuatTransDY(quat).transpose();
          else dRdQ_j = getQuatTransDZ(quat).transpose();
          
          // 链式法则: grad = Tr( (d_dis/d_Re)^T * (d_Rd/d_Qj * R_e2b) )
          Eigen::Matrix3d dRe_dQj = dRdQ_j * R_e2b;
          grad_quat(j) += weight_rot * (d_dis_d_Re.transpose() * dRe_dQj).trace();
      }

      // B. 对 Arm Thetas 的梯度 (利用反对称矩阵 [z]x)
      for (int i = 0; i < thetas.size(); ++i) {
          Eigen::Matrix4d T_i_0;
          rc_sdf_ptr_->kine_ptr_->getRelativeTransform(thetas, i, 0, T_i_0);
          Eigen::Vector3d axis_local = Eigen::Vector3d::Zero();
          axis_local(rc_sdf_ptr_->kine_ptr_->getLinks()[i+1].ang_id) = 1.0;
          
          // 1. 获取世界坐标系下的旋转轴 z_i_w
          Eigen::Vector3d z_i_w = R_drone * T_i_0.block<3,3>(0,0) * axis_local;
          
          // 2. 计算 d_Re / d_theta = [z_i_w]x * R_e
          Eigen::Matrix3d dRe_dtheta = skew_mat(z_i_w) * R_e;
          
          // 3. 链式法则: grad = Tr( (d_dis/d_Re)^T * dRe_dtheta )
          grad_thetas(i) += weight_rot * (d_dis_d_Re.transpose() * dRe_dtheta).trace();
      }
      cost += cost_rot * rhoRot;
  }

  return (cost > 1e-9);
}

// using hopf fibration:
// [a,b,c] = thrust.normalized()
// \omega_1 = sin(\phi) \dot{a] - cos(\phi) \dot{b} - (a sin(\phi) - b
// cos(\phi)) (\dot{c}/(1+c))
// \omega_2 = cos(\phi) \dot{a] - sin(\phi) \dot{b} - (a cos(\phi) - b
// sin(\phi)) (\dot{c}/(1+c))
// \omega_3 = (b \dot{a} - a \dot(b)) / (1+c)
// || \omega_12 ||^2 = \omega_1^2 + \omega_2^2 = \dot{a}^2 + \dot{b}^2 +
// \dot{c}^2

bool TrajOpt::grad_cost_omega(const Eigen::Vector3d &a,
                              const Eigen::Vector3d &j, Eigen::Vector3d &grada,
                              Eigen::Vector3d &gradj, double &cost) {
  cost = 0.0;
  grada.setZero();
  gradj.setZero();
  Eigen::Vector3d thrust_f = a - g_;
  Eigen::Vector3d zb_dot = f_DN(thrust_f) * j;
  double omega_12_sq = zb_dot.squaredNorm();
  double pen = omega_12_sq - omega_max_ * omega_max_;
  if (pen > 0) {
    double grad = 0;
    cost = smoothedL1(pen, grad);

    Eigen::Vector3d grad_zb_dot = 2 * zb_dot;
    // std::cout << "grad_zb_dot: " << grad_zb_dot.transpose() << std::endl;
    gradj = f_DN(thrust_f).transpose() * grad_zb_dot;
    grada = f_D2N(thrust_f, j).transpose() * grad_zb_dot;

    cost *= rhoOmega_;
    grad *= rhoOmega_;
    grada *= grad;
    gradj *= grad;

    return true;
  }
  return false;
}

bool TrajOpt::grad_cost_full_state(const Eigen::Vector3d &pos_ego,
                                   const Eigen::Vector4d &quat,
                                   const Eigen::VectorXd &thetas,
                                   const Eigen::Vector3d &ref_pos,
                                   const Eigen::Vector4d &ref_quat,
                                   const Eigen::VectorXd &ref_thetas,
                                   Eigen::Vector3d &gradp,
                                   Eigen::Vector4d &grad_quat,
                                   Eigen::VectorXd &grad_thetas,
                                   double &cost) {
  gradp.setZero();
  grad_quat.setZero();
  grad_thetas.resize(thetas.size());
  grad_thetas.setZero();
  cost = 0.0;

  // position cost
  Eigen::Vector3d dp = pos_ego - ref_pos;
  cost += 0.5 * rhoWP_ * dp.squaredNorm();
  gradp += rhoWP_ * dp;

  // orientation cost (rotation matrix distance)
  Eigen::Quaterniond q_b2w(quat(0), quat(1), quat(2), quat(3));
  Eigen::Matrix3d R = q_b2w.toRotationMatrix();
  Eigen::Quaterniond q_ref(ref_quat(0), ref_quat(1), ref_quat(2), ref_quat(3));
  Eigen::Matrix3d R_ref = q_ref.toRotationMatrix();
  double dis_rot = (3.0 - (R * R_ref.transpose()).trace()) / 2.0;
  double weight_rot = rhoRotFactor_ * rhoWP_;
  cost += weight_rot * dis_rot;
  Eigen::Matrix3d d_dis_d_R = -0.5 * R_ref;
  for (int j = 0; j < 4; ++j) {
    Eigen::Matrix3d dRdQ_j;
    if (j == 0)
      dRdQ_j = getQuatTransDW(quat).transpose();
    else if (j == 1)
      dRdQ_j = getQuatTransDX(quat).transpose();
    else if (j == 2)
      dRdQ_j = getQuatTransDY(quat).transpose();
    else
      dRdQ_j = getQuatTransDZ(quat).transpose();

    grad_quat(j) += weight_rot * (d_dis_d_R.transpose() * dRdQ_j).trace();
  }

  // theta cost
  if (ref_thetas.size() == thetas.size()) {
    Eigen::VectorXd dtheta = thetas - ref_thetas;
    cost += 0.5 * rhoWP_ * dtheta.squaredNorm();
    grad_thetas += rhoWP_ * dtheta;
  }

  return cost > 1e-9;
}

bool TrajOpt::grad_cost_rate(const Eigen::Vector3d &omg,
                             Eigen::Vector3d &gradomg, double &cost) {
  // ratemax: thetaDot & yawDot
  // rateframe_max: roll, pitch, actual yaw
  Eigen::VectorXd max_vec, rate_vec;
  Eigen::MatrixXd gradtmp;
  max_vec.resize(3);
  rate_vec.resize(3);
  gradtmp.resize(3, 3);
  max_vec << omega_yaw_max_, omega_max_, omega_max_;
  rate_vec << omg(2), omg(0), omg(1);
  gradtmp.col(0) << 0.0, 2 * omg(0), 0.0;
  gradtmp.col(1) << 0.0, 0.0, 2 * omg(1);
  gradtmp.col(2) << 2 * omg(2), 0.0, 0.0;
  for (int i = 0; i < 3; i++) {
    double pen = rate_vec(i) * rate_vec(i) - max_vec(i) * max_vec(i);
    if (pen > 0.0) {
      double pen2 = pen * pen;
      cost += rhoOmega_ * pen * pen2;
      gradomg += rhoOmega_ * 3 * pen2 * gradtmp.block<1, 3>(i, 0).transpose();
    }
  }
  return true;
}

bool TrajOpt::isOccupied_se3(const TrajData &traj_data, const double &cur_t) {
  Eigen::Vector3d pos, vel, acc, jer;
  pos = traj_data.getPos(cur_t);
  vel = traj_data.getVel(cur_t);
  acc = traj_data.getAcc(cur_t);
  jer = traj_data.getJer(cur_t);

  //? yaw 从哪来
  // TODO
  double yaw, dyaw;
  yaw = 0;
  dyaw = 0;

  double thr = 0.0;
  Eigen::Vector4d quat;
  Eigen::Vector3d bodyrate;

  flatmap_.forward(vel, acc, jer, yaw, dyaw, thr, quat, bodyrate);

  // TODO
  Eigen::Quaterniond q_b2w;
  q_b2w.w() = quat(3);
  q_b2w.x() = quat(0);
  q_b2w.y() = quat(1);
  q_b2w.z() = quat(2);
  // q_b2w.setIdentity();

  std::vector<Eigen::Vector3d> aabb_pts_vec, sample_pts;
  sample_pts.push_back(pos);
  gridmapPtr_->getAABBPointsSample(aabb_pts_vec, sample_pts,
                                   Eigen::Vector3d(1.0, 1.0, 1.0));

  for (int i = 0; i < aabb_pts_vec.size(); i++) {
    Eigen::Vector3d pt_body = q_b2w.inverse() * (aabb_pts_vec[i] - pos);
    Eigen::Vector3d gd;
    int box_id;
    double sdf = rc_sdf_ptr_->getDistWithGrad_body(rc_sdf_ptr_->get_thetas(),
                                                   pt_body, box_id, gd);

    // TODO 直接改成本地一个grid buffer
    // 软约束 检查时候放宽点
    if (sdf < dist_threshold_ - 0.05)
      return true;
  }

  return false;
}

inline Eigen::Matrix3d TrajOpt::Skew(const Eigen::Vector3d &v) {
  Eigen::Matrix3d S;
  S << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
  return S;
}

// 将四元数梯度(4x1)映射为轴角方向梯度(3x1)：grad_theta = J^T * grad_q
// J = 0.5 * [ -v^T ; w I + [v]_x ], where q = [w, v]
Eigen::Vector3d TrajOpt::quatGradToAxisGradient(const Eigen::Quaterniond &q,
                                                const Eigen::Vector4d &grad_q) {
  Eigen::Vector3d v(q.x(), q.y(), q.z());
  double w = q.w();

  Eigen::Matrix<double, 4, 3> J;
  J.block<1, 3>(0, 0) = -v.transpose();                            // -v^T
  J.block<3, 3>(1, 0) = w * Eigen::Matrix3d::Identity() + Skew(v); // wI + [v]_x
  J *= 0.5;

  Eigen::Vector3d grad_theta = J.transpose() * grad_q; // 3x1
  return grad_theta;
}

/**
 * @brief 使用旋转矩阵计算姿态误差及梯度
 */
void TrajOpt::calcRotationMatrixGrad(
    const Eigen::Matrix3d &R_e, const Eigen::Matrix3d &R_goal,
    const Eigen::Matrix3d &R_drone, const Eigen::VectorXd &thetas,
    double weight_rot, double &cost_rot, Eigen::VectorXd &grad_thetas) {

    // 1. 计算代价: 1 - tr(R_e * R_goal^T)/3 (归一化到 0-1 之间)
    Eigen::Matrix3d R_error = R_e * R_goal.transpose();
    cost_rot = (3.0 - R_error.trace()) / 2.0; // 类似于四元数的 1-dot^2

    // 2. 计算误差对 R_e 的导数矩阵 (3x3)
    // d_cost / d_Re = -0.5 * R_goal
    Eigen::Matrix3d d_cost_d_Re = -0.5 * R_goal;

    // 3. 遍历关节计算梯度
    for (int i = 0; i < thetas.size(); ++i) {
        // A. 获取旋转轴 z_i_w (与四元数版本一致)
        Eigen::Matrix4d T_i_0;
        rc_sdf_ptr_->kine_ptr_->getRelativeTransform(thetas, i, 0, T_i_0);
        
        int ang_id = rc_sdf_ptr_->kine_ptr_->getLinks()[i+1].ang_id;
        Eigen::Vector3d axis_local = Eigen::Vector3d::Zero();
        axis_local(ang_id) = 1.0;
        Eigen::Vector3d z_i_w = R_drone * T_i_0.block<3,3>(0,0) * axis_local;

        // B. 构造反对称矩阵 [z_i_w]x
        Eigen::Matrix3d skew_z;
        skew_z << 0, -z_i_w.z(),  z_i_w.y(),
                  z_i_w.z(), 0, -z_i_w.x(),
                 -z_i_w.y(),  z_i_w.x(), 0;

        // C. 计算 d_Re / d_theta = [z_i_w]x * R_e
        Eigen::Matrix3d d_Re_d_theta = skew_z * R_e;

        // D. 链式法则: grad = Tr( (d_cost/d_Re)^T * (d_Re/d_theta) )
        double g = (d_cost_d_Re.transpose() * d_Re_d_theta).trace();
        
        grad_thetas(i) += weight_rot * g;
    }
}

} // namespace traj_opt
