#include <traj_opt/traj_opt.h>
#include <traj_opt/lbfgs_raw.hpp>

namespace traj_opt {

// Generate MincoS3-ununiform trajectory for given traget sequence within certain prediction time
bool TrajOpt::generate_traj(const Eigen::MatrixXd& iniState,
                            const Eigen::MatrixXd& finState,
                            const double& seg_per_dis,
                            const std::vector<Eigen::Vector3d>& ego_path,
                            const std::vector<Eigen::Vector3d>& target_predcit,
                            Trajectory<5>& traj){
  //! 0. determine number of segment (a-star path raycheck-aware)
  std::vector<Eigen::Vector3d> mid_q_vec;
  extract_mid_pts_from_apath(ego_path, seg_per_dis, mid_q_vec, N_);
  visPtr_->visualize_pointcloud(mid_q_vec, "mid_waypts");
  INFO_MSG("[trajopt] Pieces: " << N_);
  INFO_MSG("0 done");

  //! 1. set opt varibles
  dim_t_ = N_ - 1;
  dim_p_ = N_ - 1;
  x_ = new double[dim_t_ + 3 * dim_p_ + 1]; // 1: delta-sumT

  Eigen::Map<Eigen::VectorXd> t(x_, dim_t_);
  Eigen::Map<Eigen::MatrixXd> P(x_ + dim_t_, 3, dim_p_);
  double& deltaT = x_[dim_t_ + dim_p_ * 3];
  sum_T_ = tracking_dur_;
  tracking_ps_ = target_predcit;

  INFO_MSG("1 done");


  //! 2. set boundary & initial value
  initS_ = iniState.block<3,3>(0,0);
  finalS_ = finState.block<3,3>(0,0);
  Eigen::VectorXd T(N_);
  T.setConstant(sum_T_ / N_);
  backwardT_sum(T, t);
  deltaT = 0.1;

  //TODO P initial value
  for (int i = 1; i < N_; ++i){
    P.col(i-1) = mid_q_vec[i-1];
  }
  INFO_MSG("2 done");


  //! 3. opt begin
  minco_s3_opt_.reset(initS_, finalS_, N_);
  lbfgs::lbfgs_parameter_t lbfgs_params;
  lbfgs::lbfgs_load_default_parameters(&lbfgs_params);
  lbfgs_params.mem_size = 16;
  lbfgs_params.past = 2;
  lbfgs_params.g_epsilon = 0;
  lbfgs_params.min_step = 1e-16;
  lbfgs_params.delta = 1e-3;
  lbfgs_params.line_search_type = 0;
  // lbfgs_params.max_linesearch = 50;
  // lbfgs_params.s_curv_coeff = 0.5;
  // lbfgs_params.f_dec_coeff = 0.5;
  double minObjective;

  int opt_ret = 0;
  auto tic = std::chrono::steady_clock::now();
  // while (cost_lock_.test_and_set())
  //   ;      
  INFO_MSG("begin to opt");
  iter_times_ = 0;
  opt_ret = lbfgs::lbfgs_optimize(dim_t_ + 3 * dim_p_ + 1, x_, &minObjective,
                                  &objectiveFuncTracking, nullptr,
                                  &earlyExitTracking, this, &lbfgs_params);
  auto toc = std::chrono::steady_clock::now();
  // cost_lock_.clear();

  std::cout << "\033[32m>ret: " << opt_ret << "\033[0m" << std::endl;
  dashboard_cost_print();
  INFO_MSG("iter: " << iter_times_);
  std::cout << "optmization costs: " << (toc - tic).count() * 1e-6 << "ms" << std::endl;

  if (opt_ret < 0) {
    delete[] x_;
    return false;
  }else{
    double sum_T = sum_T_ + deltaT * deltaT;
    forwardT_sum(t, sum_T, T);
    minco_s3_opt_.generate(P, T);
    traj = minco_s3_opt_.getTraj();
    delete[] x_;
    return true;
  }

}

inline int earlyExitTracking(void* ptrObj,
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
  if (obj.pause_debug_) {
    INFO_MSG_RED("earlyExit iter: " << obj.iter_times_);
    Eigen::Map<const Eigen::VectorXd> t(x, obj.dim_t_);
    Eigen::Map<const Eigen::MatrixXd> P(x + obj.dim_t_, 3, obj.dim_p_);
    const double& deltaT = x[obj.dim_t_ + obj.dim_p_ * 3];
    Eigen::VectorXd T(obj.N_);
    double sumT = obj.sum_T_ + deltaT * deltaT;
    forwardT_sum(t, sumT, T);

    obj.minco_s3_opt_.generate(P, T);
    auto traj = obj.minco_s3_opt_.getTraj();
    obj.visPtr_->visualize_traj(traj, "debug_traj");
    std::vector<Eigen::Vector3d> int_waypts;
    for (const auto& piece : traj) {
      const auto& dur = piece.getDuration();
      for (int i = 0; i < obj.K_; ++i) {
        double t = dur * i / obj.K_;
        int_waypts.push_back(piece.getPos(t));
      }
    }
    obj.visPtr_->visualize_pointcloud(int_waypts, "int_waypts");
    for (auto p : int_waypts){
      Eigen::Vector3d gradp;
      double costp;
      obj.grad_cost_collision(p, gradp, costp);
      double          dist = 0;
      Eigen::Vector3d dist_grad;
      dist_grad.setZero();
      // if (obj.esdfmap_.evaluateESDFWithGrad(p, dist, dist_grad)){
      //   INFO_MSG_GREEN("p:"<<p.transpose()<<", esdf: "<<dist<<", cost: "<<costp<<", grad: "<<gradp.transpose());
      // }

    }
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
inline double objectiveFuncTracking(void* ptrObj,
                                          const double* x,
                                          double* grad,
                                          const int n) {
  // std::cout << "damn" << std::endl;
  TrajOpt& obj = *(TrajOpt*)ptrObj;
  obj.iter_times_++;
  obj.clear_cost_rec();
  // INFO_MSG_RED("iter: " << obj.iter_times_);

  //! 1. fetch opt varaibles from x_
  Eigen::Map<const Eigen::VectorXd> t(x, obj.dim_t_);
  Eigen::Map<Eigen::VectorXd> gradt(grad, obj.dim_t_);

  Eigen::Map<const Eigen::MatrixXd> P(x + obj.dim_t_, 3, obj.dim_p_);
  Eigen::Map<Eigen::MatrixXd> gradP(grad + obj.dim_t_, 3, obj.dim_p_);

  const double& deltaT = x[obj.dim_t_ + obj.dim_p_ * 3];
  double& grad_deltaT = grad[obj.dim_t_ + obj.dim_p_ * 3];

  //! 2. reform P & T
  Eigen::VectorXd T(obj.N_);
  double sumT = obj.sum_T_ + deltaT * deltaT;
  forwardT_sum(t, sumT, T);

  //! 3. generate minco using P & T
  auto tic = std::chrono::steady_clock::now();
  obj.minco_s3_opt_.generate(P, T);

  double cost = obj.minco_s3_opt_.getTrajJerkCost();
  obj.minco_s3_opt_.calGrads_CT();
  obj.cost_snap_rec_ = cost;
  auto toc = std::chrono::steady_clock::now();

  //! 4. calculate penalty and gradient to C & T
  tic = std::chrono::steady_clock::now();
  obj.addTimeIntPenaltyTracking(cost);
  obj.addTimeCostTracking(cost);
  toc = std::chrono::steady_clock::now();
  // tictoc_integral_ += (toc - tic).count();

  //! 5. propogate gradient to mid-point P & T
  tic = std::chrono::steady_clock::now();
  obj.minco_s3_opt_.calGrads_PT();
  toc = std::chrono::steady_clock::now();
  // tictoc_innerloop_ += (toc - tic).count();

  //! 6. propogate gradient to opt variable P & tau
  addLayerTGrad(t, sumT, obj.minco_s3_opt_.gdT, gradt);
  gradP = obj.minco_s3_opt_.gdP;

  cost += obj.rhoT_track_ * deltaT * deltaT;
  obj.cost_t_rec_ = obj.rhoT_track_ * deltaT * deltaT;
  obj.deltaT_rec_ = deltaT * deltaT;
  grad_deltaT = (obj.minco_s3_opt_.gdT.dot(T)/sumT + obj.rhoT_track_) * 2 * deltaT;

  // obj.dashboard_cost_print();

  return cost;
}

void TrajOpt::addTimeIntPenaltyTracking(double& cost){
  // cost of one inner sample points
  double cost_inner;
  // state of innner sample point 
  Eigen::Vector3d pos, vel, acc, jer;
  // gradient of the state of innner sample point 
  Eigen::Vector3d grad_p, grad_v, grad_a;
  // derivatives of the time base
  Eigen::Matrix<double, 6, 1> beta0, beta1, beta2, beta3;
  // sample time step; alpha; trapezoidal integral weights
  double step, alpha, omg;
  // current sample time within the segment
  double st = 0.0;
  // gradient propogate to C & T of each segment
  Eigen::Matrix<double, 6, 3> gradViola_c;
  double gradViola_t;
  // exact term cost and gradient
  double cost_vel, cost_acc, cost_collision_p;
  Eigen::Vector3d grad_vel, grad_acc, grad_collision_p;

  int innerLoop = K_ + 1;


  for (int i = 0; i < N_; ++i) {
    step = minco_s3_opt_.T1(i) / K_;
    const auto& c = minco_s3_opt_.b.block<6, 3>(i * 6, 0);
    st = 0.0;

    for (int j = 0; j < innerLoop; ++j) {
      omg = (j == 0 || j == innerLoop - 1) ? 0.5 : 1.0;
      alpha = 1.0 / K_ * j;
      beta0 = cal_timebase_jerk(0, st);
      beta1 = cal_timebase_jerk(1, st);
      beta2 = cal_timebase_jerk(2, st);
      beta3 = cal_timebase_jerk(3, st);
      pos = c.transpose() * beta0;
      vel = c.transpose() * beta1;
      acc = c.transpose() * beta2;
      jer = c.transpose() * beta3;

      gradViola_c.setZero();
      gradViola_t = 0.0;
      grad_p.setZero();
      grad_v.setZero();
      grad_a.setZero();
      cost_inner = 0.0;

      if (grad_cost_collision(pos, grad_collision_p, cost_collision_p)) {
        grad_p += grad_collision_p;
        cost_inner += cost_collision_p;
        cost_collision_rec_ += omg * step * cost_collision_p;
      }
      if (grad_cost_v(vel, grad_vel, cost_vel)) {
        grad_v += grad_vel;
        cost_inner += cost_vel;
        cost_v_rec_ += omg * step * cost_vel;
      }
      if (grad_cost_a(acc, grad_acc, cost_acc)) {
        grad_a += grad_acc;
        cost_inner += cost_acc;
        cost_a_rec_ += omg * step * cost_acc;
      }

      gradViola_c = beta0 * grad_p.transpose();
      gradViola_t = grad_p.transpose() * vel;
      gradViola_c += beta1 * grad_v.transpose();
      gradViola_t += grad_v.transpose() * acc;
      gradViola_c += beta2 * grad_a.transpose();
      gradViola_t += grad_a.transpose() * jer;

      minco_s3_opt_.gdC.block<6, 3>(i * 6, 0) += omg * step * gradViola_c;
      minco_s3_opt_.gdT(i) += omg * (cost_inner / K_ + alpha * step * gradViola_t);

      cost += omg * step * cost_inner;
      st += step;
    }
  }
}

void TrajOpt::addTimeCostTracking(double& cost){
  const auto& T = minco_s3_opt_.T1;
  int M = tracking_ps_.size() * 4 / 5;
  int piece = 0;
  double t = 0;
  double t_pre = 0;

  // cost of one inner sample points
  double cost_inner;
  // state of innner sample point 
  Eigen::Vector3d pos, vel;
  // gradient of the state of innner sample point 
  Eigen::Vector3d grad_p, grad_v;
  // derivatives of the time base
  Eigen::Matrix<double, 6, 1> beta0, beta1;
  // sample time step
  double step;
  // current sample time within the segment
  double st = 0.0;
  // gradient propogate to C & T of each segment
  Eigen::Matrix<double, 6, 3> gradViola_c;
  double gradViola_t;
  // exact term cost and gradient
  double cost_tracking_p, cost_tracking_angle, cost_tracking_visibility;
  Eigen::Vector3d grad_tracking_p, grad_tracking_angle, grad_tracking_visibility;

  step = tracking_dt_;

  for (int i = 0; i < M; ++i) {
    
    // double rho = exp2(-3.0 * i / M);
    double rho = 1.0;
    while (t - t_pre > T(piece)) {
      t_pre += T(piece);
      piece++;
    }
    st = t - t_pre;
    beta0 = cal_timebase_jerk(0, st);
    beta1 = cal_timebase_jerk(1, st);
    
    const auto& c = minco_s3_opt_.b.block<6, 3>(piece * 6, 0);
    pos = c.transpose() * beta0;
    vel = c.transpose() * beta1;
    grad_p.setZero();
    grad_v.setZero();
    cost_inner = 0.0;

    Eigen::Vector3d target_p = tracking_ps_[i];

    INFO_MSG_BLUE("target_no."<<i <<", p: "<<pos.transpose());

    if (grad_cost_tracking_p(pos, target_p, grad_tracking_p, cost_tracking_p)) {
      grad_p += grad_tracking_p;
      cost_inner += cost_tracking_p;
      cost_tracking_dis_rec_ += rho * step * cost_tracking_p;
    }
    if (grad_cost_tracking_visibility(pos, target_p, grad_tracking_visibility, cost_tracking_visibility)) {
      // grad_tracking_visibility.x() = 0;
      grad_p += grad_tracking_visibility;
      cost_inner += cost_tracking_visibility;
      cost_tracking_vis_rec_ += rho * step * cost_tracking_visibility;
      INFO_MSG_RED("cost: "<<cost_tracking_visibility);
      INFO_MSG_GREEN("grad_tracking_visibility: " << grad_tracking_visibility.transpose());
    }
    // if (grad_cost_collision(pos, grad_tracking_visibility, cost_tracking_visibility)) {
    //   grad_p += grad_tracking_visibility;
    //   cost_inner += cost_tracking_visibility;
    //   cost_collision_rec_ += rho * step * cost_tracking_visibility;
    //   INFO_MSG_GREEN("grad_cost_collision: " << grad_tracking_visibility.transpose());
    // }


    INFO_MSG_GREEN("grad_p: " << grad_p.transpose());

    gradViola_c = beta0 * grad_p.transpose();
    gradViola_t = grad_p.transpose() * vel;
    minco_s3_opt_.gdC.block<6, 3>(piece * 6, 0) += rho * step * gradViola_c;
    if (piece > 0) {
      minco_s3_opt_.gdT.head(piece).array() += -rho * step * gradViola_t;
    }

    cost += rho * step * cost_inner;

    t += step;
  }
}


bool TrajOpt::grad_cost_tracking_p(const Eigen::Vector3d& p,
                                   const Eigen::Vector3d& target_p,
                                   Eigen::Vector3d& gradp,
                                   double& costp){
  double upper = tracking_dist_ + tolerance_tracking_d_;
  double lower = tracking_dist_ - tolerance_tracking_d_;
  upper = upper * upper;
  lower = lower * lower;

  Eigen::Vector3d dp = (p - target_p);
  double dr2 = dp.head(2).squaredNorm();
  double dz2 = dp.z() * dp.z();

  bool ret;
  gradp.setZero();
  costp = 0;

  double pen = dr2 - upper;
  if (pen > 0) {
    double grad;
    costp += penF(pen, grad);
    gradp.head(2) += 2 * grad * dp.head(2);
    ret = true;
  } else {
    pen = lower - dr2;
    if (pen > 0) {
      double pen2 = pen * pen;
      gradp.head(2) -= 6 * pen2 * dp.head(2);
      costp += pen2 * pen;
      ret = true;
    }
  }
  pen = dz2 - tolerance_tracking_d_ * tolerance_tracking_d_;
  if (pen > 0) {
    double pen2 = pen * pen;
    gradp.z() += 6 * pen2 * dp.z();
    costp += pen * pen2;
    ret = true;
  }

  gradp *= rhoTrackingDis_;
  costp *= rhoTrackingDis_;

  return ret;
}

bool TrajOpt::grad_cost_tracking_angle(const Eigen::Vector3d& p,
                                      const Eigen::Vector3d& target_p,
                                      Eigen::Vector3d& gradp,
                                      double& costp){
  Eigen::Vector3d a = p - target_p;
  Eigen::Vector3d b(cos(track_angle_expect_), sin(track_angle_expect_), 0.0);
  INFO_MSG_BLUE("a: " << a.transpose() << ", b: "<<b.transpose());
  double inner_product = a.dot(b);
  double norm_a = a.norm();
  double norm_b = b.norm();
  double pen = 1.0 - inner_product / norm_a / norm_b;
  INFO_MSG_BLUE("grad_cost_tracking_angle: " << inner_product / norm_a / norm_b);
  if (pen > 0) {
    double grad = 0;
    costp = smoothedL1(pen, grad);
    gradp = grad * -(norm_a * b - inner_product / norm_a * a) / norm_a / norm_a / norm_b;
    INFO_MSG_BLUE("gradp: " << gradp.transpose());

    // gradp = grad * (norm_b * cosTheta / norm_a * a - b);
    gradp *= rhoTrackingAngle_;
    costp *= rhoTrackingAngle_;
    return true;
  } else {
    return false;
  }
}

bool TrajOpt::grad_cost_tracking_visibility(const Eigen::Vector3d& p,
                                            const Eigen::Vector3d& target_p,
                                            Eigen::Vector3d& gradp,
                                            double& costp){
  costp = 0.0;
  gradp.setZero();
  double          dist;
  Eigen::Vector3d dist_grad;

  int sample = 5;
  double rho = 1.0;
  Eigen::Vector3d d = p - target_p;

  for (size_t k = 1; k < sample; k++){
    double lambda_k = k * 1.0 / sample;
    Eigen::Vector3d pk = lambda_k * p + ( 1- lambda_k ) * target_p;
    double threshold_k = rho * ( 1- lambda_k ) * d.norm();
    // double threshold_k = 2.0;


    dist = gridmapPtr_->getCostWithGrad(pk, dist_grad);

    INFO_MSG_YELLOW("pk: "<<pk.transpose() << ", rk: " << threshold_k <<", dis: "<<dist);
    INFO_MSG("disgrad: "<<dist_grad.transpose());

    if (dist < threshold_k){
      double pen = threshold_k - dist;
      double grad_L1 = 0;
      costp += pen * pen;
      gradp += 2.0 * pen * (rho * (1-lambda_k) * d / d.norm() - 
        dist_grad * lambda_k);
      // gradp += 2.0 * pen * (-dist_grad * lambda_k);
      INFO_MSG_RED("grad: " << (2.0 * pen * (-2.0 * dist * dist_grad * lambda_k)).transpose());
    }
    // if (dist < threshold_k){
    //   double pen = threshold_k - dist;
    //   double grad_L1 = 0;
    //   costp += smoothedL1(pen, grad_L1);
    //   // gradp += grad_L1 / sample * (2.0 * threshold_k * rho * (1-lambda_k) * d / d.norm() - 
    //   //   2.0 * dist * dist_grad * lambda_k);
    //   gradp += grad_L1 * (-dist_grad * lambda_k);
    //   INFO_MSG_RED("grad: " << (grad_L1 * (-2.0 * dist * dist_grad * lambda_k)).transpose());
    // }


  }
  
  // double dist_threshold = 2.0;
  // dist_grad.setZero();
  // Eigen::Vector3d pk = 0.5 * p + 0.5 * target_p;
  // dist = gridmapPtr_->getCostWithGrad(pk, dist_grad);
  // INFO_MSG_YELLOW("esdf: "<< dist<<", gra: "<<dist_grad.transpose());
  // if (dist < dist_threshold){
  //   double pen =  dist_threshold - dist;
  //   costp += pen * pen;
    
  //   gradp += -2.0 * pen * dist_grad * 0.5;
  //   INFO_MSG("p: "<<p.transpose()<<", esdf: "<< dist <<", pen: "<<pen<<", costp: "<<costp<<", grad: "<<gradp.transpose());
  // }

  gradp *= rhoTrackingVisibility_;
  costp *= rhoTrackingVisibility_;

  return true;
}

} // namespace traj_opt