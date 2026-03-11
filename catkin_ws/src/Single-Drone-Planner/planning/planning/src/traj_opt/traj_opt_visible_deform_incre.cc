#include <traj_opt/traj_opt.h>
#include <traj_opt/lbfgs_raw.hpp>

namespace traj_opt {

bool TrajOpt::generate_viewpoint(const Eigen::Vector3d& cur_tar,
                                 const Eigen::Vector3d& last_ego_p,
                                 const Eigen::Vector3d& last_tar_p,
                                 Eigen::Vector3d& viewpoint){
  dim_p_ = 1;
  x_ = new double[3]; 
  Eigen::Map<Eigen::MatrixXd> P(x_, 3, dim_p_);
  P = last_ego_p;
  cur_tar_ = cur_tar;
  last_ego_p_ = last_ego_p;
  last_tar_p_ = last_tar_p;

  //! 3. opt begin
  lbfgs::lbfgs_parameter_t lbfgs_params;
  lbfgs::lbfgs_load_default_parameters(&lbfgs_params);
  lbfgs_params.mem_size = 16;
  lbfgs_params.past = 3;
  lbfgs_params.g_epsilon = 0;
  lbfgs_params.min_step = 1e-16;
  lbfgs_params.delta = 1e-4;
  lbfgs_params.line_search_type = 0;
  double minObjective;

  int opt_ret = 0;
  auto tic = std::chrono::steady_clock::now();
  // while (cost_lock_.test_and_set())
  //   ;      
  INFO_MSG("begin to opt");
  iter_times_ = 0;
  opt_ret = lbfgs::lbfgs_optimize(3 * dim_p_, x_, &minObjective,
                                  &objectiveFuncDeform, nullptr,
                                  &earlyExitDeform, this, &lbfgs_params);
  auto toc = std::chrono::steady_clock::now();
  // cost_lock_.clear();

  std::cout << "\033[32m>ret: " << opt_ret << "\033[0m" << std::endl;
  dashboard_cost_print();
  INFO_MSG("iter: " << iter_times_);
  std::cout << "optmization costs: " << (toc - tic).count() * 1e-6 << "ms" << std::endl;

  int a;
  std::cin >> a;

  if (opt_ret < 0) {
    delete[] x_;
    return false;
  }else{
    viewpoint = P;
    delete[] x_;
    return true;
  }
}


bool TrajOpt::visible_path_deform(const Eigen::Vector3d& ego_p, 
                                  const std::vector<Eigen::Vector3d>& target_predcit,
                                  std::vector<Eigen::Vector3d>& viewpoint_path){
  viewpoint_path.clear();
  Eigen::Vector3d last_ego_p = ego_p;
  Eigen::Vector3d last_tar_p = target_predcit.front();
  for (size_t i = 1; i < target_predcit.size(); i++){
    Eigen::Vector3d cur_tar = target_predcit[i];
    Eigen::Vector3d viewpoint;
    if (generate_viewpoint(cur_tar, last_ego_p, last_tar_p, viewpoint)){
      viewpoint_path.push_back(viewpoint);
    }else{
      INFO_MSG_RED("generate_viewpoint ERROR");
    }
    last_ego_p = viewpoint;
    last_tar_p = cur_tar;
  }
  return true;
}


inline int earlyExitDeform(void* ptrObj,
                          const double* x,
                          const double* grad,
                          const double fx,
                          const double xnorm,
                          const double gnorm,
                          const double step,
                          int n,
                          int k,
                          int ls) {
  TrajOpt& obj = *(TrajOpt*)ptrObj;
  INFO_MSG_RED("earlyExit iter: " << obj.iter_times_);
  if (obj.pause_debug_) {
    Eigen::Map<const Eigen::MatrixXd> P(x, 3, obj.dim_p_);
    std::vector<Eigen::Vector3d> int_waypts;
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> rays;
    for (size_t i = 0; i < obj.dim_p_; i++){
      int_waypts.push_back(P.col(i));
      rays.emplace_back(P.col(i), obj.cur_tar_);
    }
    obj.visPtr_->visualize_pointcloud(int_waypts, "int_waypts");
    obj.visPtr_->visualize_pairline(rays, "visible_pts");

    INFO_MSG_YELLOW("in process");
    obj.dashboard_cost_print();
    int a;
    std::cin >> a;

    // NOTE pause
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  return 0;
}


// SECTION object function
inline double objectiveFuncDeform(void* ptrObj,
                                  const double* x,
                                  double* grad,
                                  const int n) {
  // std::cout << "damn" << std::endl;
  TrajOpt& obj = *(TrajOpt*)ptrObj;
  obj.iter_times_++;
  obj.clear_cost_rec();
  INFO_MSG_RED("iter: " << obj.iter_times_);

  //! 1. fetch opt varaibles from x_
  Eigen::Map<const Eigen::MatrixXd> P(x, 3, obj.dim_p_);
  Eigen::Map<Eigen::MatrixXd> gradP(grad, 3, obj.dim_p_);

  //! 2. calculate cost and grad for each term
  double cost = 0.0;
  double cost_p = 0.0;
  Eigen::Vector3d grad_p;
  grad_p.setZero();

  obj.addDeformCost(P, obj.cur_tar_, grad_p, cost_p);
  cost += cost_p;
  grad_p.z() = 0.0;
  gradP += grad_p;

  // costDistance
  // const double dEps = 1e-2;
  // obj.cost_snap_rec_ = 0;
  // for (size_t i = 1; i < obj.dim_p_; i++){
  //   Eigen::Vector3d pa = P.col(i-1);
  //   Eigen::Vector3d pb = P.col(i);
  //   Eigen::Vector3d d  = pb - pa;
  //   double dis = sqrt(d.squaredNorm() + dEps);

  //   cost += dis;
  //   obj.cost_snap_rec_ += dis;
  //   gradP.col(i) += d / dis;
  //   gradP.col(i-1) += -d / dis;
  // }

  // costAngle
  // const double angEps = 1e-4;
  // obj.cost_snap_rec_ = 0;

  // Eigen::Vector3d pa = obj.last_ego_p_;
  // Eigen::Vector3d pb = P;
  // Eigen::Vector3d ta = obj.last_tar_p_;
  // Eigen::Vector3d tb = obj.cur_tar_;
  // Eigen::Vector3d a  = pa - ta;
  // Eigen::Vector3d b  = pb - tb;
  // double inner_product = a.dot(b);
  // double norm_a = a.norm() + angEps;
  // double norm_b = b.norm() + angEps;
  // double pen = 0.98 - inner_product / norm_a / norm_b;

  // if (pen > 0){
  //   INFO_MSG_RED("a: "<<a.transpose()<<", b: "<<b.transpose()<<", pen: "<<pen);
  //   cost += obj.rhoDeformConsistAngle_ * pen;
  //   obj.cost_snap_rec_ += obj.rhoDeformConsistAngle_ * pen;
  //   gradP += obj.rhoDeformConsistAngle_ * -(norm_b * a - inner_product / norm_b * b) / norm_b / norm_b / norm_a;
  // }
  

  return cost;
}

void TrajOpt::addDeformCost(const Eigen::Vector3d& p,
                            const Eigen::Vector3d& target_p,
                            Eigen::Vector3d& gradp,
                            double& costp){
  gradp.setZero();
  costp = 0.0;
  // exact term cost and gradient
  double cost_tracking_p, cost_tracking_angle, cost_tracking_visibility;
  Eigen::Vector3d grad_tracking_p, grad_tracking_angle, grad_tracking_visibility;
  
  if (grad_cost_deform_dis(p, target_p, grad_tracking_p, cost_tracking_p)) {
    gradp += grad_tracking_p;
    costp += cost_tracking_p;
    cost_tracking_dis_rec_ += cost_tracking_p;
  }
  if (grad_cost_deform_angle(p, target_p, grad_tracking_angle, cost_tracking_angle)) {
    gradp += grad_tracking_angle;
    costp += cost_tracking_angle;
    cost_tracking_ang_rec_ += cost_tracking_angle;
  }
  if (grad_cost_deform_visibility(p, target_p, grad_tracking_visibility, cost_tracking_visibility)) {
    gradp += grad_tracking_visibility;
    costp += cost_tracking_visibility;
    cost_tracking_vis_rec_ += cost_tracking_visibility;
  }
}

bool TrajOpt::grad_cost_deform_dis(const Eigen::Vector3d& p,
                                  const Eigen::Vector3d& target_p,
                                  Eigen::Vector3d& gradp,
                                  double& costp){
  const double dEps = 1e-2;
  gradp.setZero();
  Eigen::Vector3d dp = (p - target_p);
  double pen = dp.head(2).norm() - tracking_dist_;
  costp = pen * pen;
  gradp.head(2) = 2.0 * pen * dp.head(2) / (dp.head(2).norm() + dEps);
  
  gradp *= rhoDeformDis_;
  costp *= rhoDeformDis_;

  return true;
}

bool TrajOpt::grad_cost_deform_angle(const Eigen::Vector3d& p,
                                     const Eigen::Vector3d& target_p,
                                     Eigen::Vector3d& gradp,
                                     double& costp){
  const double angEps = 1e-4;
  Eigen::Vector3d a = p - target_p;
  Eigen::Vector3d b(cos(track_angle_expect_), sin(track_angle_expect_), 0.0);
  INFO_MSG_BLUE("a: " << a.transpose() << ", b: "<<b.transpose());
  double inner_product = a.dot(b);
  double norm_a = a.norm() + angEps;
  double norm_b = b.norm() + angEps;
  double pen = 1.0 - inner_product / norm_a / norm_b;
  INFO_MSG_BLUE("grad_cost_tracking_angle: " << inner_product / norm_a / norm_b);
  if (pen > 0) {
    double grad = 0;
    costp = smoothedL1(pen, grad);
    gradp = grad * -(norm_a * b - inner_product / norm_a * a) / norm_a / norm_a / norm_b;
    // INFO_MSG_BLUE("gradp: " << gradp.transpose());

    // gradp = grad * (norm_b * cosTheta / norm_a * a - b);
    gradp *= rhoDeformAngle_;
    costp *= rhoDeformAngle_;
    return true;
  } else {
    return false;
  }
}

bool TrajOpt::grad_cost_deform_visibility(const Eigen::Vector3d& p,
                                          const Eigen::Vector3d& target_p,
                                          Eigen::Vector3d& gradp,
                                          double& costp){
  costp = 0.0;
  gradp.setZero();
  double          dist;
  Eigen::Vector3d dist_grad;

  int sample = 10;
  double rho = 1.0;
  Eigen::Vector3d d = p - target_p;

  for (size_t k = 0; k < sample; k++){
    double lambda_k = k * 1.0 / sample;
    Eigen::Vector3d pk = lambda_k * p + ( 1- lambda_k ) * target_p;
    double threshold_k_vis = 0.8;
    double threshold_k_collision = 0.2;

    dist = gridmapPtr_->getCostWithGrad(pk, dist_grad);

    // INFO_MSG_YELLOW("pk: "<<pk.transpose() << ", rk: " << threshold_k_vis <<", dis: "<<dist);

    if (dist < threshold_k_vis){
      double pen = threshold_k_vis * threshold_k_vis - dist * dist;
      double grad_L1 = 0;
      costp += rhoDeformVisibility_/sample * smoothedL1(pen, grad_L1);
      gradp += rhoDeformVisibility_/sample * grad_L1 * (-2.0 * dist * dist_grad * lambda_k);
    }

    if (dist < threshold_k_collision){
      double pen = threshold_k_collision * threshold_k_collision - dist * dist;
      double grad_L1 = 0;
      costp += rhoDeformP_/sample * smoothedL1(pen, grad_L1);
      gradp += rhoDeformP_/sample * grad_L1 * (-2.0 * dist * dist_grad * lambda_k);
    }

  }

  return true;
}

} // namespace traj_opt