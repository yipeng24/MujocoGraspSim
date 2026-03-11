#include <traj_opt/traj_opt.h>
#include <traj_opt/lbfgs_raw.hpp>

namespace traj_opt {

// Generate MincoS3(none-uniform) trajectory for given goal
bool TrajOpt::generate_traj(const Eigen::MatrixXd& iniState,
                            const Eigen::MatrixXd& finState,
                            const double& seg_per_dis,
                            const std::vector<Eigen::Vector3d>& ego_path,
                            Trajectory<5>& traj){

  if ((iniState.col(0) - finState.col(0)).norm() < gridmapPtr_->resolution() * 2.0) return false;
  //! 0. determine number of segment (a-star path raycheck-aware)
  std::vector<Eigen::Vector3d> mid_q_vec;
  extract_mid_pts_from_apath(ego_path, seg_per_dis, mid_q_vec, N_);
  // visPtr_->visualize_pointcloud(mid_q_vec, "mid_waypts");
  INFO_MSG("[trajopt] Pieces: " << N_);
  INFO_MSG("0 done");
  
  //! 1. set opt varibles
  dim_t_ = N_;
  dim_p_ = N_ - 1;
  x_ = new double[dim_t_ + 3 * dim_p_]; 
  Eigen::Map<Eigen::VectorXd> t(x_, dim_t_);
  Eigen::Map<Eigen::MatrixXd> P(x_ + dim_t_, 3, dim_p_);

  INFO_MSG("1 done");


  //! 2. set boundary & initial value
  initS_ = iniState;
  finalS_ = finState;
  // P initial value
  for (int i = 1; i < N_; ++i){
    P.col(i-1) = mid_q_vec[i-1];
  }
  Eigen::VectorXd T(N_);
  INFO_MSG("2.1 done");
  get_init_taj(N_, P, T);
  INFO_MSG("2.2 done");
  backwardT(T, t);
  INFO_MSG("2 done");

  


  //! 3. opt begin
  minco_s3_opt_.reset(iniState.block<3,3>(0,0), finState.block<3,3>(0,0), N_);
  lbfgs::lbfgs_parameter_t lbfgs_params;
  lbfgs::lbfgs_load_default_parameters(&lbfgs_params);
  lbfgs_params.mem_size = 16;
  lbfgs_params.past = 3;
  lbfgs_params.g_epsilon = 0;
  lbfgs_params.min_step = 1e-32;
  lbfgs_params.delta = 1e-4;
  lbfgs_params.line_search_type = 0;
  double minObjective;

  int opt_ret = 0;
  auto tic = std::chrono::steady_clock::now();
  // while (cost_lock_.test_and_set())
  //   ;      
  INFO_MSG("begin to opt");
  iter_times_ = 0;
  opt_ret = lbfgs::lbfgs_optimize(dim_t_ + 3 * dim_p_, x_, &minObjective,
                                  &objectiveFuncGoal, nullptr,
                                  &earlyExitGoal, this, &lbfgs_params);
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
    forwardT(t, T);
    minco_s3_opt_.generate(P, T);
    traj = minco_s3_opt_.getTraj();
    delete[] x_;
    return true;
  }

}

inline int earlyExitGoal(void* ptrObj,
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
    Eigen::VectorXd T(obj.N_);
    forwardT(t, T);

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
      // double          dist = 0;
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
inline double objectiveFuncGoal(void* ptrObj,
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

  //! 2. reform P & T
  Eigen::VectorXd T(obj.N_);
  forwardT(t, T);

  //! 3. generate minco using P & T
  auto tic = std::chrono::steady_clock::now();
  obj.minco_s3_opt_.generate(P, T);

  double cost = obj.minco_s3_opt_.getTrajJerkCost();
  obj.minco_s3_opt_.calGrads_CT();
  obj.cost_snap_rec_ = cost;
  auto toc = std::chrono::steady_clock::now();

  //! 4. calculate penalty and gradient to C & T
  tic = std::chrono::steady_clock::now();
  obj.addTimeIntPenaltyGoal(cost);
  // obj.addTimeCostTracking(cost);
  toc = std::chrono::steady_clock::now();
  // tictoc_integral_ += (toc - tic).count();

  //! 5. propogate gradient to mid-point P & T
  tic = std::chrono::steady_clock::now();
  obj.minco_s3_opt_.calGrads_PT();
  toc = std::chrono::steady_clock::now();
  // tictoc_innerloop_ += (toc - tic).count();

  //! 6. propogate gradient to opt variable P & tau
  obj.minco_s3_opt_.gdT.array() += obj.rhoT_track_;
  cost += obj.rhoT_track_ * T.sum();
  obj.cost_t_rec_ = obj.rhoT_track_ * T.sum();

  addLayerTGrad(t, obj.minco_s3_opt_.gdT, gradt);
  gradP = obj.minco_s3_opt_.gdP;

  // obj.dashboard_cost_print();

  return cost;
}

void TrajOpt::addTimeIntPenaltyGoal(double& cost){
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

} // namespace traj_opt