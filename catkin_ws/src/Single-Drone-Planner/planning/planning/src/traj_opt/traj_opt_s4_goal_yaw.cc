#include <traj_opt/traj_opt.h>
#include <traj_opt/lbfgs_raw.hpp>

namespace traj_opt {
using rot_util = rotation_util::RotUtil;

// Generate MincoS3(none-uniform) trajectory for given goal
bool TrajOpt::generate_traj(const Eigen::MatrixXd& iniState,
                            const Eigen::MatrixXd& finState,
                            const Eigen::MatrixXd& init_yaw,
                            const Eigen::MatrixXd& final_yaw,
                            const double& seg_per_dis,
                            const std::vector<Eigen::Vector3d>& ego_path,
                            Trajectory<7>& traj){

  if ((iniState.col(0) - finState.col(0)).norm() < gridmapPtr_->resolution() * 2.0) 
  {
    ROS_INFO("reach goal, no need to generate traj");
    return false;
  }
  //! 0. determine number of segment (a-star path raycheck-aware)
  std::vector<Eigen::Vector3d> mid_q_vec;
  extract_mid_pts_from_apath(ego_path, seg_per_dis, mid_q_vec, N_);
  visPtr_->visualize_pointcloud(mid_q_vec, "mid_waypts");

  inter_wps_ = mid_q_vec;
  inter_wps_.push_back(ego_path.back());
  

  // N_ = ego_path.size() - 1;
  INFO_MSG("[trajopt] Pieces: " << N_);
  INFO_MSG("0 done");
  
  //! 1. set opt varibles
  dim_t_ = N_;
  dim_p_ = N_ - 1;
  x_ = new double[dim_t_ + 4 * dim_p_]; 
  Eigen::Map<Eigen::VectorXd> t(x_, dim_t_);
  Eigen::Map<Eigen::MatrixXd> P(x_ + dim_t_, 3, dim_p_);
  Eigen::Map<Eigen::VectorXd> yaw(x_ + dim_t_+ 3 * dim_p_, dim_p_);
  INFO_MSG("1 done");

  //! 2. set boundary & initial value
  initS_ = iniState;
  finalS_ = finState;
  init_yaw_.resize(1, 2);
  final_yaw_.resize(1, 2);
  init_yaw_ = init_yaw;
  final_yaw_ = final_yaw;
  double error_angle = rot_util::error_angle(init_yaw(0, 0), final_yaw_(0, 0));
  final_yaw_(0, 0) = init_yaw_(0, 0) + error_angle;

  // P initial value
  for (int i = 1; i < N_; ++i){
    P.col(i-1) = mid_q_vec[i - 1];
  }
  Eigen::VectorXd T(N_);
  INFO_MSG("2.1 done");
  get_init_s4_taj(N_, P, T);
  INFO_MSG("2.2 done");
  backwardT(T, t);
  INFO_MSG("2 done");
  


  //! 3. opt begin
  minco_s4_opt_.reset(iniState.block<3,4>(0,0), finState.block<3,4>(0,0), N_);

  // get yaw init value
  // average
  // for (int i = 0; i < N_-1; ++i){
  //   yaw(i) = init_yaw(0, 0) + (i / N_) * error_angle;
  // }
  // look forward
  minco_s4_opt_.generate(P, T);
  auto traj_info = minco_s4_opt_.getTraj();
  for(int i = 1; i < N_; ++i){
    Eigen::Vector2d vel_i = traj_info.getJuncVel(i).head(2);
    yaw(i - 1) = atan2(vel_i(1), vel_i(0));
    if(i > 1){
      error_angle = rot_util::error_angle(yaw(i - 2), yaw(i - 1));
      yaw(i - 1) = yaw(i - 2) + error_angle;
    }else{
      error_angle = rot_util::error_angle(init_yaw_(0,0), yaw(i - 1));
      yaw(i - 1) = init_yaw_(0,0) + error_angle;
    }
  }
  error_angle = rot_util::error_angle(yaw(N_ - 2), final_yaw_(0, 0));
  final_yaw_(0, 0) = yaw(N_ - 2) + error_angle;
  INFO_MSG_RED("init_yaw: " << init_yaw_(0,0)<<", end_yaw: " << final_yaw_(0,0));


  #ifdef USE_RC_SDF

    // method 1 获取前1s轨迹 采样点 aabb
    std::vector<Eigen::Vector3d> sample_pts;
    double t_end = 1.0 > traj_info.getTotalDuration() ? traj_info.getTotalDuration() : 1.0;
    double dt = 0.2;
    // 希望加入
    for (double t = 0; t < t_end; t += dt){ //遍历轨迹
      Eigen::Vector3d p = traj_info.getPos(t);
      sample_pts.push_back(p);
    }
    sample_pts.push_back(traj_info.getPos(t_end));
    gridmapPtr_->getAABBPoints(aabb_pts_,sample_pts);

    // method 2 获取轨迹完整aabb
    // gridmapPtr_->getAABBPoints(aabb_pts_,iniState.col(0), finState.col(0));

  #endif

  minco_s2_yaw_opt_.reset(init_yaw_, final_yaw_, N_);
  lbfgs::lbfgs_parameter_t lbfgs_params;
  lbfgs::lbfgs_load_default_parameters(&lbfgs_params);
  lbfgs_params.mem_size = 16;
  lbfgs_params.past = 3;
  // lbfgs_params.g_epsilon = 0;
  // lbfgs_params.min_step = 1e-32;
  lbfgs_params.delta = 1e-4;
  lbfgs_params.line_search_type = 1;
  double minObjective;

  int opt_ret = 0;
  auto tic = std::chrono::steady_clock::now();
  // while (cost_lock_.test_and_set())
  //   ;      
  INFO_MSG("begin to opt");
  iter_times_ = 0;
  opt_ret = lbfgs::lbfgs_optimize(dim_t_ + 4 * dim_p_, x_, &minObjective,
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
    minco_s4_opt_.generate(P, T);
    minco_s2_yaw_opt_.generate(yaw, T);
    traj = getS4TrajWithYaw(minco_s4_opt_, minco_s2_yaw_opt_);
    
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
    Eigen::Map<const Eigen::MatrixXd> yaw(x + obj.dim_t_ + obj.dim_p_ * 3, 1, obj.dim_p_);

    Eigen::VectorXd T(obj.N_);
    forwardT(t, T);

    obj.minco_s4_opt_.generate(P, T);
    obj.minco_s2_yaw_opt_.generate(yaw, T);

    auto traj = getS4TrajWithYaw(obj.minco_s4_opt_, obj.minco_s2_yaw_opt_);
    INFO_MSG_GREEN("traj type: " << traj.getType());
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

    // std::vector<Eigen::Vector3d> aabb_pts_vec;
    // for(int i=0; i<obj.aabb_pts_.cols();++i)
    //   aabb_pts_vec.push_back(obj.aabb_pts_.col(i));
    // obj.visPtr_->visualize_pointcloud(aabb_pts_vec, "aabb_pts");

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
  Eigen::Map<const Eigen::VectorXd>  t(x, obj.dim_t_);
  Eigen::Map<Eigen::VectorXd> gradt(grad, obj.dim_t_);

  Eigen::Map<const Eigen::MatrixXd> P(x + obj.dim_t_, 3, obj.dim_p_);
  Eigen::Map<Eigen::MatrixXd> gradP(grad + obj.dim_t_, 3, obj.dim_p_);

  Eigen::Map<const Eigen::MatrixXd> yaw(x + obj.dim_t_ + obj.dim_p_ * 3, 1, obj.dim_p_);
  Eigen::Map<Eigen::MatrixXd> grad_yaw(grad + obj.dim_t_ + obj.dim_p_ * 3, 1, obj.dim_p_);

  //! 2. reform P & T
  Eigen::VectorXd T(obj.N_);
  forwardT(t, T);

  //! 3. generate minco using P & T
  auto tic = std::chrono::steady_clock::now();
  obj.minco_s4_opt_.generate(P, T);
  obj.minco_s2_yaw_opt_.generate(yaw, T);

  double cost = obj.minco_s4_opt_.getTrajSnapCost();
  obj.minco_s4_opt_.calGrads_CT();
  obj.cost_snap_rec_ = cost;

  cost += obj.minco_s2_yaw_opt_.getEnergyCost();
  obj.minco_s2_yaw_opt_.calGrads_CT();
  auto toc = std::chrono::steady_clock::now();

  //! 4. calculate penalty and gradient to C & T
  tic = std::chrono::steady_clock::now();
  obj.addTimeIntPenaltyGoal(cost);
  toc = std::chrono::steady_clock::now();
  // tictoc_integral_ += (toc - tic).count();

  //! 5. propogate gradient to mid-point P & T
  tic = std::chrono::steady_clock::now();
  obj.minco_s4_opt_.calGrads_PT();
  obj.minco_s2_yaw_opt_.calGrads_PT();
  toc = std::chrono::steady_clock::now();
  // tictoc_innerloop_ += (toc - tic).count();

  //! 6. propogate gradient to opt variable P & tau
  obj.minco_s4_opt_.gdT.array() += obj.rhoT_;
  cost += obj.rhoT_ * T.sum();
  obj.cost_t_rec_ = obj.rhoT_ * T.sum();

  addLayerTGrad(t, obj.minco_s4_opt_.gdT + obj.minco_s2_yaw_opt_.gdT, gradt);
  gradP = obj.minco_s4_opt_.gdP;
  grad_yaw = obj.minco_s2_yaw_opt_.gdP;

  obj.dashboard_cost_print();

  return cost;
}

void TrajOpt::addTimeIntPenaltyGoal(double& cost){
  // cost of one inner sample points
  double cost_inner;
  // state of innner sample point 
  Eigen::Vector3d pos, vel, acc, jer, sna;
  double yaw, dyaw, ddyaw;
  // gradient of the state of innner sample point 
  Eigen::Vector3d grad_p, grad_v, grad_a, grad_j;
  double grad_yaw, grad_dyaw;
  // derivatives of the time base
  Eigen::Matrix<double, 8, 1> beta0, beta1, beta2, beta3, beta4;
  Eigen::Matrix<double, 4, 1> beta_yaw0, beta_yaw1, beta_yaw2;
  // sample time step; alpha; trapezoidal integral weights
  double step, alpha, omg;
  // current sample time within the segment
  double st = 0.0;
  // gradient propogate to C & T of each segment
  Eigen::Matrix<double, 8, 3> gradViola_c;
  double gradViola_t;
  Eigen::Matrix<double, 4, 1> gradViola_c_yaw;
  double gradViola_t_yaw;
  // exact term cost and gradient
  double cost_vel, cost_acc, cost_collision_p, cost_vel_yaw, cost_yaw;
  Eigen::Vector3d grad_vel, grad_acc, grad_collision_p;
  double grad_vel_yaw;
  double grad_pos_yaw;

  int innerLoop = K_ + 1;

  // ROS_WARN("!!!!!!!!!!!!!n= %d", N_);
  for (int i = 0; i < N_; ++i) {
    step = minco_s4_opt_.T1(i) / K_;
    const auto& c = minco_s4_opt_.b.block<8, 3>(i * 8, 0);
    const auto& c_yaw = minco_s2_yaw_opt_.b.block<4, 1>(i * 4, 0);
    st = 0.0;

    for (int j = 0; j < innerLoop; ++j) {
      omg = (j == 0 || j == innerLoop - 1) ? 0.5 : 1.0;
      alpha = 1.0 / K_ * j;
      beta0 = cal_timebase_snap(0, st);
      beta1 = cal_timebase_snap(1, st);
      beta2 = cal_timebase_snap(2, st);
      beta3 = cal_timebase_snap(3, st);
      beta4 = cal_timebase_snap(4, st);
      beta_yaw0 = cal_timebase_acc(0, st);
      beta_yaw1 = cal_timebase_acc(1, st);
      beta_yaw2 = cal_timebase_acc(2, st);
      pos   = c.transpose() * beta0;
      vel   = c.transpose() * beta1;
      acc   = c.transpose() * beta2;
      jer   = c.transpose() * beta3;
      sna   = c.transpose() * beta4;
      yaw   = c_yaw.transpose() * beta_yaw0;
      dyaw  = c_yaw.transpose() * beta_yaw1;
      ddyaw = c_yaw.transpose() * beta_yaw2;

      gradViola_c.setZero();
      gradViola_t = 0.0;
      grad_p.setZero();
      grad_v.setZero();
      grad_a.setZero();
      grad_j.setZero();
      gradViola_c_yaw.setZero();
      gradViola_t_yaw = 0.0;
      grad_yaw = 0.0;
      grad_dyaw = 0.0;
      cost_inner = 0.0;

    #ifdef USE_ESDF

      if (grad_cost_collision(pos, grad_collision_p, cost_collision_p)) {
        grad_p += grad_collision_p;
        cost_inner += cost_collision_p;
        cost_collision_rec_ += omg * step * cost_collision_p;
      }

    #elif defined(USE_RC_SDF)

      // ROS_ERROR("--------- USE_RC_SDF --------");
      // std::cout << "n_pts:" << aabb_pts_.cols() << std::endl;
      for(int idx = 0; idx < aabb_pts_.cols(); ++idx){
        if (grad_cost_collision_rc(pos, aabb_pts_.col(idx), grad_collision_p, cost_collision_p)) {
          grad_p += grad_collision_p;
          cost_inner += cost_collision_p;
          cost_collision_rec_ += omg * step * cost_collision_p;
        }
      }
      // ROS_ERROR("--------- USE_RC_SDF --------");

    #endif


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
      if (grad_cost_dyaw(dyaw, grad_vel_yaw, cost_vel_yaw)) {
        grad_dyaw += grad_vel_yaw;
        cost_inner += cost_vel_yaw;
        cost_dyaw_rec_ += omg * step * cost_vel_yaw;
      }
      if(i > 1 && i < N_ - 1){ // ignore head and tail piece
        if(grad_cost_yaw_forward(yaw, vel, grad_pos_yaw, grad_vel, cost_yaw)) {
          grad_yaw += grad_pos_yaw;
          grad_v += grad_vel;
          cost_inner += cost_yaw;
          cost_yaw_rec_ += omg * step * cost_yaw;
        }
      }
      gradViola_c = beta0 * grad_p.transpose();
      gradViola_t = grad_p.transpose() * vel;
      gradViola_c += beta1 * grad_v.transpose();
      gradViola_t += grad_v.transpose() * acc;
      gradViola_c += beta2 * grad_a.transpose();
      gradViola_t += grad_a.transpose() * jer;
      gradViola_c += beta3 * grad_j.transpose();
      gradViola_t += grad_j.transpose() * sna;

      gradViola_c_yaw = beta_yaw0 * grad_yaw;
      gradViola_t_yaw = grad_yaw * dyaw;
      gradViola_c_yaw += beta_yaw1 * grad_dyaw;
      gradViola_t_yaw += grad_dyaw * ddyaw;

      minco_s4_opt_.gdC.block<8, 3>(i * 8, 0) += omg * step * gradViola_c;
      minco_s4_opt_.gdT(i) += omg * (cost_inner / K_ + alpha * step * gradViola_t + alpha * step * gradViola_t_yaw);

      minco_s2_yaw_opt_.gdC.block<4, 1>(i * 4, 0) += omg * step * gradViola_c_yaw;

      cost += omg * step * cost_inner;
      st += step;
    }
  }
}
} // namespace traj_opt