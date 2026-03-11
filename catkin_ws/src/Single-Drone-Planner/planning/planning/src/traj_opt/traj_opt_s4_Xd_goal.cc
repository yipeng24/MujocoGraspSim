#include <traj_opt/traj_opt.h>
#include <traj_opt/lbfgs_raw.hpp>

namespace traj_opt {

static double tictoc_innerloop_;
static double tictoc_integral_;

// Generate MincoS3(none-uniform) trajectory for given goal
bool TrajOpt::generate_traj_clutter(const Eigen::MatrixXd& iniState,
                            const Eigen::MatrixXd& finState,
                            const Eigen::MatrixXd& init_yaw,
                            const Eigen::MatrixXd& final_yaw,
                            const Eigen::MatrixXd& init_thetas,
                            const Eigen::MatrixXd& final_thetas,
                            const double& seg_per_dis,
                            const std::vector<Eigen::VectorXd>& ego_pathXd,
                            Trajectory<7>& traj){

  INFO_MSG_GREEN_BG("----- Start gen traj");

  // if ((iniState.col(0) - finState.col(0)).norm() < gridmapPtr_->resolution()) 
  if ((iniState.col(0) - finState.col(0)).norm() < 0.1) 
  {
    ROS_INFO("reach goal, no need to generate traj");
    return false;
  }
  clear_cost_rec();

  double yaw_out;
  Eigen::Vector3d pos_out;
  final_thetas_.resize(n_thetas_,2);
  final_thetas_.setZero();
  final_thetas_ = final_thetas;

  initS_ = iniState;
  finalS_ = finState;
  init_yaw_.resize(1, 2);
  final_yaw_.resize(1, 2);
  init_yaw_ = init_yaw;
  final_yaw_ = final_yaw;

  //! get P initial value  
  std::vector<Eigen::Vector3d> mid_q_vec,ego_path;
  std::vector<Eigen::VectorXd> mid_q_vecXd, mid_q_thetas;
  std::vector<double> mid_q_yaw;
  for (const auto& ptXd : ego_pathXd){
    ego_path.push_back(ptXd.head(3));
    std::cout << "ptXd: " << ptXd.transpose() << std::endl;
  }
  // extract_mid_pts_from_apath(ego_path, seg_per_dis, mid_q_vec, N_);
  extract_mid_pts_from_apath(ego_pathXd, seg_per_dis, mid_q_vecXd, N_);
  INFO_MSG_GREEN_BG("[GenTraj] 1 Extract mid points done");
  
  for (const auto& mid_q : mid_q_vecXd){
    mid_q_vec.push_back(mid_q.head(3));
    mid_q_yaw.push_back(mid_q(3));
    mid_q_thetas.push_back(mid_q.tail(mid_q.size()-4));
    // std::cout << "mid_q: " << mid_q_vec.back().transpose() << std::endl;
    // std::cout << "mid_q_yaw: " << mid_q_yaw.back() << std::endl;
    // std::cout << "mid_q_thetas: " << mid_q_thetas.back().transpose() << std::endl;
  }
  visPtr_->visualize_pointcloud(mid_q_vec, "mid_waypts");
  INFO_MSG_GREEN_BG("[GenTraj] 2 feed mid_q_vec done");

  //! set opt varibles
  dim_t_ = N_;
  dim_p_ = N_ - 1;
  x_ = new double[dim_t_ + 4 * dim_p_ + n_thetas_ * dim_p_]; 
  Eigen::Map<Eigen::VectorXd> t(x_, dim_t_);
  Eigen::Map<Eigen::MatrixXd> P(x_ + dim_t_, 3, dim_p_);
  Eigen::Map<Eigen::VectorXd> yaw(x_ + dim_t_+ 3 * dim_p_, dim_p_);
  std::vector<Eigen::Map<Eigen::VectorXd>> thetas_vec;
  for (int i = 0; i < n_thetas_; ++i)
    thetas_vec.push_back(Eigen::Map<Eigen::VectorXd>(x_ + dim_t_+ (4+i) * dim_p_, dim_p_));
  INFO_MSG_GREEN_BG("[GenTraj] 3 Set opt varibles done");

  // N_ = ego_path.size() - 1;
  INFO_MSG("[trajopt] Pieces: " << N_);

  // std::cout << "P initial value: " << std::endl;
  for (int i = 0; i < N_-1; ++i){
    // std::cout << "P.rows: " << P.rows() << " P.cols: " << P.cols() << std::endl;
    // std::cout << "mid_q_vec[i].rows: " << mid_q_vec[i].rows() << std::endl;
    P.col(i) = mid_q_vec[i];
  }
  Eigen::VectorXd T(N_);
  get_init_s4_taj(N_, P, T);
  backwardT(T, t);

  minco_s4_opt_.reset(iniState.block<3,4>(0,0), finState.block<3,4>(0,0), N_);
  minco_s4_opt_.generate(P, T);
  auto traj_info = minco_s4_opt_.getTraj();

  //! get yaw initial value
  INFO_MSG_GREEN_BG("[GenTraj] start feed yaw");
  double error_angle_yaw;
  // 1 get yaw init value
  // average
  // for (int i = 0; i < N_-1; ++i){
  //   yaw(i) = init_yaw(0, 0) + ((i+1) / N_) * error_angle_yaw;
  // }
  
  // 2 look forward
  // for(int i = 1; i < N_; ++i){
  //   Eigen::Vector2d vel_i = traj_info.getJuncVel(i).head(2);
  //   yaw(i - 1) = atan2(vel_i(1), vel_i(0));
  //   if(i > 1){
  //     error_angle_yaw = rot_util::error_angle(yaw(i - 2), yaw(i - 1));
  //     yaw(i - 1) = yaw(i - 2) + error_angle_yaw;
  //   }else{
  //     error_angle_yaw = rot_util::error_angle(init_yaw_(0,0), yaw(i - 1));
  //     yaw(i - 1) = init_yaw_(0,0) + error_angle_yaw;
  //   }
  // }
  // error_angle_yaw = rot_util::error_angle(yaw(N_ - 2), final_yaw_(0, 0));
  // final_yaw_(0, 0) = yaw(N_ - 2) + error_angle_yaw;

  // 3 w/ init yaws
  for(int i=0; i<N_-1; ++i){
    std::cout << "----- i: " << i << std::endl;
    if(i==0){
      std::cout << "init_yaw: " << init_yaw_(0,0) << std::endl;
      // std::cout << "mid_q_yaw[i]: " << mid_q_yaw[i] << std::endl;
      error_angle_yaw = rot_util::error_angle(init_yaw_(0,0), mid_q_yaw[i]);
      yaw(i) = init_yaw_(0,0) + error_angle_yaw;
      // std::cout << "yaw(" << i << "): " << yaw(i) << std::endl;
    }else if(i==N_-2){
      // std::cout << "final_yaw: " << final_yaw_(0,0) << std::endl;
      // std::cout << "mid_q_yaw[i-1]: " << mid_q_yaw[i-1] << std::endl;
      error_angle_yaw = rot_util::error_angle(mid_q_yaw[i-1], final_yaw_(0,0));
      yaw(i) = mid_q_yaw[i-1] + error_angle_yaw;
      // std::cout << "yaw(" << i << "): " << yaw(i) << std::endl;
    }else{
      // std::cout << "mid_q_yaw[i-1]: " << mid_q_yaw[i-1] << std::endl;
      // std::cout << "mid_q_yaw[i]: " << mid_q_yaw[i] << std::endl;
      error_angle_yaw = rot_util::error_angle(mid_q_yaw[i-1], mid_q_yaw[i]);
      yaw(i) = mid_q_yaw[i-1] + error_angle_yaw;
      // std::cout << "yaw(" << i << "): " << yaw(i) << std::endl;
    }
  }
  error_angle_yaw = rot_util::error_angle(yaw(N_ - 2), final_yaw_(0, 0));
  final_yaw_(0, 0) = yaw(N_ - 2) + error_angle_yaw;

  minco_s2_yaw_opt_.reset(init_yaw_, final_yaw_, N_);
  
  INFO_MSG_GREEN_BG("[GenTraj] 4 yaw init done");
  INFO_MSG_RED("yaw_in: " << init_yaw_(0,0)<<", yaw_out: " << final_yaw_(0,0));
  for (int i = 0; i < N_-1; ++i)
    std::cout << "yaw(" << i << "): " << yaw(i) << std::endl;

  //! get thetas initial value
  init_thetas_.resize(n_thetas_,2);
  init_thetas_ = init_thetas;
  Eigen::VectorXd error_angle_thetas(n_thetas_);

  // 1 linear insert
  // for (int i = 0; i < n_thetas_; ++i){
  //   error_angle_thetas(i) = rot_util::error_angle(init_thetas_(i, 0), final_thetas_(i, 0));
  //   final_thetas_(i, 0) = init_thetas_(i, 0) + error_angle_thetas(i);
  //   for(int j = 0; j < N_-1; ++j){
  //     thetas_vec[i](j) = init_thetas_(i, 0) + ((double)(j+1) / (double)(N_)) * error_angle_thetas(i);
  //   }
  //   minco_s2_theta_opt_vec_[i].reset(init_thetas_.block(i,0,1,2), final_thetas_.block(i,0,1,2), N_);
  // }

  // 2 w/ init thetas
  for (int i = 0; i < n_thetas_; ++i){
    for(int j = 0; j < N_-1; ++j)
      if(j==0){
        error_angle_thetas(i) = rot_util::error_angle(init_thetas_(i, 0), mid_q_thetas[j](i));
        thetas_vec[i](j) = init_thetas_(i, 0) + error_angle_thetas(i);
      }else if(j==N_-2){
        error_angle_thetas(i) = rot_util::error_angle(mid_q_thetas[j-1](i), final_thetas_(i, 0));
        thetas_vec[i](j) = mid_q_thetas[j-1](i) + error_angle_thetas(i);
      }else{
        error_angle_thetas(i) = rot_util::error_angle(mid_q_thetas[j-1](i), mid_q_thetas[j](i));
        thetas_vec[i](j) = mid_q_thetas[j-1](i) + error_angle_thetas(i);
      }
    minco_s2_theta_opt_vec_[i].reset(init_thetas_.block(i,0,1,2), final_thetas_.block(i,0,1,2), N_);
  }

  INFO_MSG_GREEN_BG("[GenTraj] 5 thetas init done");
  std::cout << "thetas_in: \n" << init_thetas_.transpose() << std::endl;
  std::cout << "thetas_out: \n" << final_thetas_.transpose() << std::endl;
  for (int i = 0; i < n_thetas_; ++i)
    std::cout << "thetas(" << i << "): " << thetas_vec[i].transpose() << std::endl;

  //! vis initial value



  //! get obs pts
  #ifdef USE_RC_SDF

    std::vector<Eigen::Vector3d> sample_pts;
    double t_end = traj_info.getTotalDuration();
    // double t_end = traj_info.getTotalDuration() > 3.0 ? 3.0 :traj_info.getTotalDuration();
    double dt = 0.2;
    // 希望加入
    for (double t = 0; t < t_end; t += dt){ //遍历轨迹
      Eigen::Vector3d p = traj_info.getPos(t);
      sample_pts.push_back(p);
    }
    sample_pts.push_back(traj_info.getPos(t_end));
    gridmapPtr_->getAABBPointsSample(aabb_pts_vec_,sample_pts);

    visPtr_->visualize_pointcloud(aabb_pts_vec_, "AABB_pts");

    INFO_MSG_GREEN_BG("[GenTraj] 4 Get obs pts done, pts_num:" << aabb_pts_vec_.size());

  #endif

  //! init flatmap params
  Eigen::VectorXd physicalParams(6);
  // params from px4 control
  physicalParams(0) = 0.61;
  physicalParams(1) = 9.8;
  physicalParams(2) = 0.10;
  physicalParams(3) = 0.23;
  physicalParams(4) = 0.01;
  physicalParams(5) = 0.02;  
  flatmap_.reset(physicalParams(0), physicalParams(1), physicalParams(2),
                      physicalParams(3), physicalParams(4), physicalParams(5));

  INFO_MSG_GREEN_BG("[GenTraj] 1 Params set done");


  //! begin to opt
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
  tictoc_innerloop_ = 0;
  tictoc_integral_ = 0;
  // while (cost_lock_.test_and_set())
  //   ;      
  INFO_MSG_GREEN("Begin to opt");
  iter_times_ = 0;
  opt_ret = lbfgs::lbfgs_optimize(dim_t_ + ( 4 + n_thetas_ )* dim_p_, x_, &minObjective,
                                  &objectiveFuncGoal, nullptr,
                                  &earlyExitGoal, this, &lbfgs_params);
  auto toc = std::chrono::steady_clock::now();
  // cost_lock_.clear();

  INFO_MSG_GREEN_BG("[GenTraj] 5 Opt done");

  std::cout << "innerloop costs: " << tictoc_innerloop_ * 1e-6 << "ms" << std::endl;
  std::cout << "integral costs: " << tictoc_integral_ * 1e-6 << "ms" << std::endl;

  std::cout << "\033[32m>ret: " << opt_ret << "\033[0m" << std::endl;
  dashboard_cost_print();
  INFO_MSG("iter: " << iter_times_);
  std::cout << "optmization costs: " << (toc - tic).count() * 1e-6 << "ms" << std::endl;

  // if (opt_ret < 0) {
  //   delete[] x_;
  //   return false;
  // }else{

    forwardT(t, T);
    minco_s4_opt_.generate(P, T);
    minco_s2_yaw_opt_.generate(yaw, T);

    for(int i = 0; i < n_thetas_; ++i){
      minco_s2_theta_opt_vec_[i].generate(thetas_vec[i], T);
    }
    // auto traj_theta0 = minco_s2_theta_opt_vec_[0].getTraj();

    // traj = getS4TrajWithYaw(minco_s4_opt_, minco_s2_yaw_opt_);

    traj = getS4TrajWithYawAndThetas(minco_s4_opt_, minco_s2_yaw_opt_, minco_s2_theta_opt_vec_);

    // double total_time = traj.getTotalDuration();
    // std::cout << "total time: " << total_time << std::endl;
    // for(double tt = 0; tt < total_time; tt += 0.1){
    //   std::cout << "---time: " << tt << std::endl;
    //   std::cout << "thetas: " << traj.getTheta(tt).transpose() << std::endl;
    //   std::cout << "tt:" << tt << std::endl;
    //   // std::cout << "thetas2: " << traj_theta0.getAngle(tt) << std::endl;
    //   std::cout << "pos: " << traj.getPos(tt).transpose() << std::endl;
    //   std::cout << "vel: " << traj.getVel(tt).transpose() << std::endl;
    //   std::cout << "yaw:" << traj.getAngle(tt) << std::endl;
    //   std::cout << "--------" << std::endl;
    // }

    delete[] x_;

    return opt_ret>0;
  // }
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
    std::vector<Eigen::Map<const Eigen::VectorXd>> thetas_vec;
    for (int i = 0; i < obj.n_thetas_; ++i)
      thetas_vec.push_back(Eigen::Map<const Eigen::VectorXd>(x + obj.dim_t_ + (4+i) * obj.dim_p_, obj.dim_p_));

    Eigen::VectorXd T(obj.N_);
    forwardT(t, T);

    obj.minco_s4_opt_.generate(P, T);
    obj.minco_s2_yaw_opt_.generate(yaw, T);
    for(int i = 0; i < obj.n_thetas_; ++i){
      obj.minco_s2_theta_opt_vec_[i].generate(thetas_vec[i], T);
    }

    auto traj = getS4TrajWithYawAndThetas(obj.minco_s4_opt_, obj.minco_s2_yaw_opt_, obj.minco_s2_theta_opt_vec_);
    INFO_MSG_GREEN("traj type: " << traj.getType());
    // INFO_MSG_GREEN_BG("earlyExit 1");

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

    // INFO_MSG_GREEN_BG("earlyExit 2");
    //! check grad
    Eigen::VectorXd grad_arm_angles;
    Eigen::Vector4d grad_quat, grad_collision_quat;
    double grad_pos_yaw;
    double cost_collision_p, costp;
    Eigen::Vector3d grad_collision_p, gradp;

    std::vector<std::pair<Eigen::Vector3d,Eigen::Vector3d>> gd_p_arr, gd_q_arr;
    std::vector<Eigen::VectorXd> pathXd, pathXd_col;

    // INFO_MSG_GREEN_BG("earlyExit 3");

    for (const auto& piece : traj) {
      
      Eigen::VectorXd ptXd(4+obj.minco_s2_theta_opt_vec_.size());

      gradp.setZero();
      costp = 0;
      grad_quat.setZero();

      const auto& dur = piece.getDuration();

      // INFO_MSG_GREEN("earlyExit 3.1");

      for (int i = 0; i < obj.K_; ++i) {
        double t = dur * i / obj.K_;
        Eigen::Vector3d pos = piece.getPos(t);
        Eigen::Vector3d vel = piece.getVel(t);
        Eigen::Vector3d acc = piece.getAcc(t);
        Eigen::Vector3d jer = piece.getJer(t);
        double yaw = piece.getAngle(t);
        double dyaw = piece.getAngleRate(t);
        Eigen::VectorXd thetas = piece.getTheta(t);

        // INFO_MSG_GREEN("earlyExit 3.2");

        Eigen::Vector4d quat;
        Eigen::Vector3d br;
        double thr;
        obj.flatmap_.forward(vel, acc, jer, yaw, dyaw, thr, quat, br);
        Eigen::Quaterniond q_b2w(quat[0], quat[1], quat[2], quat[3]);

        // INFO_MSG_GREEN("earlyExit 3.3");

        for(int idx = 0; idx < obj.aabb_pts_vec_.size(); ++idx){
          if(obj.grad_sdf_full_state(obj.aabb_pts_vec_[idx], pos, quat, thetas, yaw, 
            grad_collision_p, grad_collision_quat, grad_pos_yaw, grad_arm_angles, cost_collision_p))
          {
            gradp+=grad_collision_p;
            grad_quat+=grad_collision_quat;
            costp+=cost_collision_p;
          }
        }

        // INFO_MSG_GREEN("earlyExit 3.4");

        ptXd.head(3) = pos;
        ptXd(3) = yaw;
        ptXd.tail(obj.minco_s2_theta_opt_vec_.size()) = thetas;

        if(costp > DBL_EPSILON){
          gd_p_arr.push_back(std::make_pair(pos,pos+costp*gradp));
          pathXd_col.push_back(ptXd);
          // gd_q_arr.push_back(std::make_pair(quat,grad_quat));
        }

        pathXd.push_back(ptXd);
      }
    }
    obj.visPtr_->visualize_arrows(gd_p_arr, "debug_grad_collision_p");
    obj.visPtr_->visualize_arrows(gd_q_arr, "debug_grad_collision_q");
    obj.rc_sdf_ptr_->visRobotSeq(pathXd_col, "debug_grad_collision_rob",visualization_rc_sdf::Color::red);
    obj.rc_sdf_ptr_->visRobotSeq(pathXd, "debug_robot_seq",visualization_rc_sdf::Color::grey);

    // std::vector<Eigen::Vector3d> aabb_pts_vec;
    // for(int i=0; i<obj.aabb_pts_.cols();++i)
    //   aabb_pts_vec.push_back(obj.aabb_pts_.col(i));
    // obj.visPtr_->visualize_pointcloud(aabb_pts_vec, "aabb_pts");

    INFO_MSG_YELLOW("in process");
    obj.dashboard_cost_print();
    int a;
    std::cin >> a;

    // NOTE pause
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
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

  int n_thetas = obj.minco_s2_theta_opt_vec_.size();

  //! 1. fetch opt varaibles from x_
  Eigen::Map<const Eigen::VectorXd>  t(x, obj.dim_t_);
  Eigen::Map<Eigen::VectorXd> gradt(grad, obj.dim_t_);

  Eigen::Map<const Eigen::MatrixXd> P(x + obj.dim_t_, 3, obj.dim_p_);
  Eigen::Map<Eigen::MatrixXd> gradP(grad + obj.dim_t_, 3, obj.dim_p_);

  Eigen::Map<const Eigen::MatrixXd> yaw(x + obj.dim_t_ + obj.dim_p_ * 3, 1, obj.dim_p_);
  Eigen::Map<Eigen::MatrixXd> grad_yaw(grad + obj.dim_t_ + obj.dim_p_ * 3, 1, obj.dim_p_);

  std::vector<Eigen::Map<const Eigen::MatrixXd>> theta_vec;
  std::vector<Eigen::Map<Eigen::MatrixXd>> grad_theta_vec;
  for(int i = 0; i < n_thetas; ++i){
    theta_vec.push_back(Eigen::Map<const Eigen::MatrixXd>(x + obj.dim_t_ + (4+i) * obj.dim_p_, 1, obj.dim_p_));
    grad_theta_vec.push_back(Eigen::Map<Eigen::MatrixXd>(grad + obj.dim_t_ + (4+i) * obj.dim_p_, 1, obj.dim_p_));
  }

  INFO_MSG_GREEN("objectiveFuncGoal 1.0 ");

  std::cout << "P: " << std::endl;
  for (int i = 0; i < obj.dim_p_; ++i) {
    std::cout << P.col(i).transpose() << std::endl;
  }
  std::cout << "t: " << t.transpose() << std::endl;
  std::cout << "yaw: " << yaw.transpose() << std::endl;
  for (int i = 0; i < n_thetas; ++i) {
    std::cout << "theta_" << i << ": " << theta_vec[i].transpose() << std::endl;
  }

  INFO_MSG_GREEN("objectiveFuncGoal 1.1 ");

  //! 2. reform P & T
  Eigen::VectorXd T(obj.N_);
  forwardT(t, T);

  //! 3. generate minco using P & T
  auto tic = std::chrono::steady_clock::now();
  obj.minco_s4_opt_.generate(P, T);
  obj.minco_s2_yaw_opt_.generate(yaw, T);
  for(int i = 0; i < n_thetas; ++i){
    obj.minco_s2_theta_opt_vec_[i].generate(theta_vec[i], T);
  }

  // INFO_MSG_GREEN("objectiveFuncGoal 2.0 ");

  double cost = obj.minco_s4_opt_.getTrajSnapCost();
  obj.minco_s4_opt_.calGrads_CT();
  obj.cost_snap_rec_ = cost;

  cost += obj.minco_s2_yaw_opt_.getEnergyCost();
  obj.minco_s2_yaw_opt_.calGrads_CT();

  for(int i = 0; i < n_thetas; ++i){
    obj.minco_s2_theta_opt_vec_[i].getEnergyCost();
    obj.minco_s2_theta_opt_vec_[i].calGrads_CT();
  }
  auto toc = std::chrono::steady_clock::now();

  // INFO_MSG_GREEN("objectiveFuncGoal 3.0 ");

  //! 4. calculate penalty and gradient to C & T
  tic = std::chrono::steady_clock::now();
  obj.addTimeIntPenaltyGoal(cost);
  toc = std::chrono::steady_clock::now();
  tictoc_integral_ += (toc - tic).count();

  // INFO_MSG_GREEN("objectiveFuncGoal 4.0 ");

  //! 5. propogate gradient to mid-point P & T
  tic = std::chrono::steady_clock::now();
  obj.minco_s4_opt_.calGrads_PT();
  obj.minco_s2_yaw_opt_.calGrads_PT();
  for(int i = 0; i < n_thetas; ++i){
    obj.minco_s2_theta_opt_vec_[i].calGrads_PT();
  }
  toc = std::chrono::steady_clock::now();
  tictoc_innerloop_ += (toc - tic).count();
  // INFO_MSG_GREEN("objectiveFuncGoal 5.0 ");

  //! 6. propogate gradient to opt variable P & tau
  obj.minco_s4_opt_.gdT.array() += obj.rhoT_;
  cost += obj.rhoT_ * T.sum();
  obj.cost_t_rec_ = obj.rhoT_ * T.sum();

  auto gdT = obj.minco_s4_opt_.gdT;
  gdT += obj.minco_s2_yaw_opt_.gdT;
  for(int i = 0; i < n_thetas; ++i)
    gdT += obj.minco_s2_theta_opt_vec_[i].gdT;
  // INFO_MSG_GREEN("objectiveFuncGoal 6.0 ");

  addLayerTGrad(t, gdT, gradt);
  gradP = obj.minco_s4_opt_.gdP;
  grad_yaw = obj.minco_s2_yaw_opt_.gdP;
  for(int i = 0; i < n_thetas; ++i)
    grad_theta_vec[i] = obj.minco_s2_theta_opt_vec_[i].gdP;
  // INFO_MSG_GREEN("objectiveFuncGoal 7.0 ");

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
  Eigen::Vector3d grad_p, grad_v, grad_a, grad_j, grad_omega;
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
  // total gradient
  Eigen::Vector3d totalGradPos, totalGradVel, totalGradAcc, totalGradJer;
  double totalGradPsi = 0.0, totalGradPsiD = 0.0;
  // collision related
  double thr = 0.0;
  Eigen::Vector4d quat;
  Eigen::Vector3d bodyrate;
  Eigen::Vector4d grad_quat, grad_collision_quat;
  // exact term cost and gradient
  double cost_rate = 0.0;
  Eigen::Vector3d grad_rate;
  double grad_thr;
  // arm angles
  int n_thetas = minco_s2_theta_opt_vec_.size();
  Eigen::VectorXd thetas(n_thetas), dthetas(n_thetas), ddthetas(n_thetas);
  Eigen::VectorXd grad_thetas(n_thetas), grad_dthetas(n_thetas);
  std::vector<Eigen::Matrix<double, 4, 1>> gradViola_c_theta_vec(n_thetas);
  Eigen::VectorXd gradViola_t_thetas(n_thetas);
  Eigen::VectorXd grad_vel_thetas(n_thetas), grad_collision_thetas(n_thetas);
  Eigen::VectorXd cost_vel_thetas(n_thetas);

  int innerLoop = K_ + 1;

  // ROS_WARN("!!!!!!!!!!!!!n= %d", N_);
  for (int i = 0; i < N_; ++i) {
    step = minco_s4_opt_.T1(i) / K_;
    const auto& c = minco_s4_opt_.b.block<8, 3>(i * 8, 0);
    const auto& c_yaw = minco_s2_yaw_opt_.b.block<4, 1>(i * 4, 0);
    std::vector<Eigen::Matrix<double, 4, 1>> c_theta_vec;
    for(size_t j=0; j<n_thetas; ++j)
      c_theta_vec.push_back(minco_s2_theta_opt_vec_[j].b.block<4, 1>(i * 4, 0));

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

      for(size_t k=0; k<n_thetas; ++k){
        thetas(k) = c_theta_vec[k].transpose() * beta_yaw0;
        dthetas(k) = c_theta_vec[k].transpose() * beta_yaw1;
        ddthetas(k) = c_theta_vec[k].transpose() * beta_yaw2;
        
        // double theta_temp, dtheta_temp, ddtheta_temp;
        // theta_temp = c_theta_vec[k].transpose() * beta_yaw0;
      }

      

      // INFO_MSG_GREEN("addTimeIntPenaltyGoal(" << i << "," << j << ") 1.0 ");

      //! isOccupied_se3 也调用了
      flatmap_.forward(vel, acc, jer, yaw, dyaw, thr, quat, bodyrate);

      gradViola_c.setZero();
      gradViola_t = 0.0;
      grad_p.setZero();
      grad_v.setZero();
      grad_a.setZero();
      grad_j.setZero();
      grad_yaw = 0.0;
      grad_dyaw = 0.0;
      grad_quat.setZero();
      grad_omega.setZero();
      grad_thr = 0; //no use
      grad_thetas.setZero();
      grad_dthetas.setZero();
      grad_vel_thetas.setZero();
      cost_inner = 0.0;
    

      #ifdef USE_ESDF

        if (grad_cost_collision(pos, grad_collision_p, cost_collision_p)) {
          grad_p += grad_collision_p;
          cost_inner += cost_collision_p;
          cost_collision_rec_ += omg * step * cost_collision_p;
        }

      #elif defined(USE_RC_SDF)
        // std::cout << "before grad_cost_collision_rc" << std::endl;
        // std::cout << "aabb_pts_vec_.size():" << aabb_pts_vec_.size() << std::endl;
        // INFO_MSG("box_num:" << rc_sdf_ptr_->getBoxNum());
        for(int idx = 0; idx < aabb_pts_vec_.size(); ++idx){
          
          //! Ver: pickle
          // if (grad_cost_collision_rc(pos, aabb_pts_vec_[idx], grad_collision_p, cost_collision_p)) {
          //   grad_p += grad_collision_p;
          //   cost_inner += cost_collision_p;
          //   cost_collision_rec_ += omg * step * cost_collision_p;
          // }

          //! Ver: SE3 w/ arm
          Eigen::VectorXd arm_angles; // TODO
          arm_angles = rc_sdf_ptr_->get_thetas();
          if(grad_sdf_full_state(aabb_pts_vec_[idx], pos, quat, arm_angles, yaw, 
            grad_collision_p, grad_collision_quat, grad_pos_yaw, grad_collision_thetas, cost_collision_p))
          {
            grad_p += grad_collision_p;
            grad_quat += grad_collision_quat;
            grad_thetas += grad_collision_thetas;

            // std::cout << "grad_collision_p:" << grad_collision_p.transpose() << std::endl;
            // std::cout << "grad_collision_quat:" << grad_collision_quat.transpose() << std::endl;
            // std::cout << "grad_collision_thetas:" << grad_collision_thetas.transpose() << std::endl;
            // std::cout << "cost_collision_p:" << cost_collision_p << std::endl;

            cost_inner += cost_collision_p;
            cost_collision_rec_ += omg * step * cost_collision_p;
          }
        }

        // std::cout << "after grad_cost_collision_rc" << std::endl;

        Eigen::Vector3d grad_tmp, grad_tmp2;
        double cost_tmp;
        if (grad_cost_omega(acc, jer, grad_tmp, grad_tmp2, cost_tmp)) {
          grad_a += grad_tmp;
          grad_j += grad_tmp2;
          cost_inner += cost_tmp;
          cost_omega_rec_ += omg * step * cost_tmp;
        }

        // TODO: fix me
        // if (grad_cost_rate(bodyrate, grad_rate, cost_rate)){
        //   grad_omega += grad_rate;
        //   cost_inner += cost_rate;
        //   cost_omega_rec_ += omg * step * cost_rate;
        // }

        for(size_t k=0; k<n_thetas; ++k){
          if (grad_cost_dyaw(dthetas(k), grad_vel_thetas(k), cost_vel_thetas(k))) {
            grad_dthetas(k) += grad_vel_thetas(k);
            cost_inner += cost_vel_thetas(k);
            cost_dthetas_rec_(k) += omg * step * cost_vel_thetas(k);
          }
        }

        // INFO_MSG_GREEN("addTimeIntPenaltyGoal(" << i << "," << j << ") 2.0 ");


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

      // gradViola_c = beta0 * grad_p.transpose();
      // gradViola_t = grad_p.transpose() * vel;
      // gradViola_c += beta1 * grad_v.transpose();
      // gradViola_t += grad_v.transpose() * acc;
      // gradViola_c += beta2 * grad_a.transpose();
      // gradViola_t += grad_a.transpose() * jer;
      // gradViola_c += beta3 * grad_j.transpose();
      // gradViola_t += grad_j.transpose() * sna;

      // gradViola_c_yaw = beta_yaw0 * grad_yaw;
      // gradViola_t_yaw = grad_yaw * dyaw;
      // gradViola_c_yaw += beta_yaw1 * grad_dyaw;
      // gradViola_t_yaw += grad_dyaw * ddyaw;

      flatmap_.backward(grad_p, grad_v, grad_a, grad_thr, grad_quat, grad_omega,
                        totalGradPos, totalGradVel, totalGradAcc, totalGradJer,
                        totalGradPsi, totalGradPsiD);

      totalGradPsi += grad_yaw;
      totalGradPsiD += grad_dyaw;

      gradViola_c = beta0 * totalGradPos.transpose();
      gradViola_t = totalGradPos.transpose() * vel;
      gradViola_c += beta1 * totalGradVel.transpose();
      gradViola_t += totalGradVel.transpose() * acc;
      gradViola_c += beta2 * totalGradAcc.transpose();
      gradViola_t += totalGradAcc.transpose() * jer;
      gradViola_c += beta3 * totalGradJer.transpose();
      gradViola_t += totalGradJer.transpose() * sna;

      gradViola_c_yaw = beta_yaw0 * totalGradPsi;
      gradViola_t_yaw = totalGradPsi * dyaw;
      gradViola_c_yaw += beta_yaw1 * totalGradPsiD;
      gradViola_t_yaw += totalGradPsiD * ddyaw;

      // INFO_MSG_GREEN("addTimeIntPenaltyGoal(" << i << "," << j << ") 2.5 ");

      for(size_t k=0; k<n_thetas; ++k){
        
        // std::cout << "k1:" << k << std::endl;

        gradViola_c_theta_vec[k] = beta_yaw0 * grad_thetas(k);
        // gradViola_t_thetas(k) = grad_thetas(k) * dthetas(k);
        gradViola_c_theta_vec[k] += beta_yaw1 * grad_dthetas(k);
        // gradViola_t_thetas(k) += grad_dthetas(k) * ddthetas(k);

        // std::cout << "k2:" << k << std::endl;
        //? 为啥这也加在 yaw 上
        gradViola_t_yaw += grad_thetas(k) * dthetas(k);
        gradViola_t_yaw += grad_dthetas(k) * ddthetas(k);
      }

      // INFO_MSG_GREEN("addTimeIntPenaltyGoal(" << i << "," << j << ") 3.0 ");

      minco_s4_opt_.gdC.block<8, 3>(i * 8, 0) += omg * step * gradViola_c;
      minco_s4_opt_.gdT(i) += omg * (cost_inner / K_ + alpha * step * gradViola_t 
                            + alpha * step * gradViola_t_yaw);

      minco_s2_yaw_opt_.gdC.block<4, 1>(i * 4, 0) += omg * step * gradViola_c_yaw;

      for(size_t k=0; k<n_thetas; ++k){
        minco_s2_theta_opt_vec_[k].gdC.block<4, 1>(i * 4, 0) += omg * step * gradViola_c_theta_vec[k];
      }

      // INFO_MSG_GREEN("addTimeIntPenaltyGoal(" << i << "," << j << ") 4.0 ");

      cost += omg * step * cost_inner;
      st += step;
    }
  }

}
} // namespace traj_opt