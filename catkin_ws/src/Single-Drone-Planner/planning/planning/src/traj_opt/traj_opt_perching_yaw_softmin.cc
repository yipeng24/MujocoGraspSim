#include <traj_opt/traj_opt.h>

#include <traj_opt/lbfgs_raw.hpp>

namespace traj_opt {
using rot_util = rotation_util::RotUtil;

static Eigen::Vector3d car_p_, car_v_;
static double car_theta_, car_omega_;
static double v_max_horiz_ = 0;
static Eigen::Vector3d tail_q_v_;
static double theta_d_, theta_f_; // the landing platform normal vector angle wrt z-axis and x-y plane, respectively
static Eigen::Vector3d g_(0, 0, -9.8);
static Eigen::Vector3d land_v_;
static Eigen::Vector3d v_t_x_, v_t_y_;
static Trajectory<7> init_traj_;
static double init_tail_f_;
static Eigen::Vector2d init_vt_, init_at_;
static bool initial_guess_ = false;
static bool is_last_succ_ = true;

static double thrust_middle_, thrust_half_;

static double tictoc_innerloop_;
static double tictoc_integral_;

static double expect_traj_duration_;

static Eigen::MatrixXd tailS_;

static bool q2v(const Eigen::Quaterniond& q,
                Eigen::Vector3d& v) {
  Eigen::MatrixXd R = q.toRotationMatrix();
  v = R.col(2).normalized();
  return true;
}
static Eigen::Vector3d f_N(const Eigen::Vector3d& x) {
  return x.normalized();
}
static Eigen::MatrixXd f_DN(const Eigen::Vector3d& x) {
  double x_norm_2 = x.squaredNorm();
  return (Eigen::MatrixXd::Identity(3, 3) - x * x.transpose() / x_norm_2) / sqrt(x_norm_2);
}
static Eigen::MatrixXd f_D2N(const Eigen::Vector3d& x, const Eigen::Vector3d& y) {
  double x_norm_2 = x.squaredNorm();
  double x_norm_3 = x_norm_2 * x.norm();
  Eigen::MatrixXd A = (3 * x * x.transpose() / x_norm_2 - Eigen::MatrixXd::Identity(3, 3));
  return (A * y * x.transpose() - x * y.transpose() - x.dot(y) * Eigen::MatrixXd::Identity(3, 3)) / x_norm_3;
}

// SECTION  variables transformation and gradient transmission


static double forward_thrust(const double& f) {
  return thrust_half_ * sin(f) + thrust_middle_;
  // return f;
}
static void addLayerThrust(const double& f,
                           const double& grad_thrust,
                           double& grad_f) {
  grad_f = thrust_half_ * cos(f) * grad_thrust;
  // grad_f = grad_thrust;
}

static void forwardTailV(const Eigen::Ref<const Eigen::Vector2d>& xy,
                         Eigen::Ref<Eigen::Vector3d> tailV) {
  tailV = land_v_ + xy.x() * v_t_x_ + xy.y() * v_t_y_;
}

static double softmin(const double& f1, const double& f2, double& grad_f1, double& grad_f2){
  static double alpha = -1e-4;
  double sum_exp = exp(alpha * f1) + exp(alpha * f2);
  double S = 1 / alpha * log(sum_exp);
  INFO_MSG_YELLOW("f1: " << f1 << ", f2: " << f2 <<", sume: " << sum_exp << ", S: " << S);
  grad_f1 = exp(alpha * f1) / sum_exp;
  grad_f2 = exp(alpha * f2) / sum_exp;
  return S;
}

// ref: Vehicle Trajectory Prediction based on Motion Model and Maneuver Recognition
static void CYRV_model(const Eigen::Vector3d& p0, const Eigen::Vector3d& v0, const double& theta, const double& omega, const double& t,
                       Eigen::Vector3d& p1, Eigen::Vector3d& gradt)
{
  if (initial_guess_ && expect_traj_duration_ < 2.0 && omega > 1e-2){
    double v = v0.head(2).dot(Eigen::Vector2d(cos(theta), sin(theta)));
    p1.x() = p0.x() + v / omega * sin(theta + omega*t) - v / omega *sin(theta);
    p1.y() = p0.y() - v / omega * cos(theta + omega*t) + v / omega *cos(theta);
    p1.z() = p0.z();
    gradt.x() = v * cos(theta + omega*t);
    gradt.y() = v * sin(theta + omega*t);
    gradt.z() = 0.0;
  }else{
    p1.x() = p0.x() + v0.x() * t;
    p1.y() = p0.y() + v0.y() * t;
    p1.z() = p0.z();// + v0.z() * t;
    gradt.x() = v0.x();
    gradt.y() = v0.y();
    // gradt.z() = v0.z();
    gradt.z() = 0;
  }
}


// !SECTION variables transformation and gradient transmission

static double getMaxOmega(Trajectory<7>& traj) {
  double dt = 0.01;
  double max_omega = 0;
  for (double t = 0; t < traj.getTotalDuration(); t += dt) {
    Eigen::Vector3d a = traj.getAcc(t);
    Eigen::Vector3d j = traj.getJer(t);
    Eigen::Vector3d thrust = a - g_;
    Eigen::Vector3d zb_dot = f_DN(thrust) * j;
    double omega12 = zb_dot.norm();
    if (omega12 > max_omega) {
      max_omega = omega12;
    }
  }
  return max_omega;
}

static void bvp(const double& t,
                const Eigen::MatrixXd i_state,
                const Eigen::MatrixXd f_state,
                Piece<7>::CoefficientMat& coeffMat) {
  double t1 = t;
  double t2 = t1 * t1;
  double t3 = t2 * t1;
  double t4 = t2 * t2;
  double t5 = t3 * t2;
  double t6 = t3 * t3;
  double t7 = t4 * t3;
  Piece<7>::CoefficientMat boundCond;
  std::cout << "i_state: " << i_state.rows()<<", "<<i_state.cols()  << std::endl;
  std::cout << "f_state: " << f_state.rows()<<", "<<f_state.cols()  << std::endl;
  
  boundCond.leftCols(4) = i_state;
  boundCond.rightCols(4) = f_state;

  std::cout << "boundCond: " << boundCond.rows()<<", "<<boundCond.cols()  << std::endl;


  coeffMat.col(0) = (boundCond.col(7) / 6.0 + boundCond.col(3) / 6.0) * t3 +
                    (-2.0 * boundCond.col(6) + 2.0 * boundCond.col(2)) * t2 +
                    (10.0 * boundCond.col(5) + 10.0 * boundCond.col(1)) * t1 +
                    (-20.0 * boundCond.col(4) + 20.0 * boundCond.col(0));
  coeffMat.col(1) = (-0.5 * boundCond.col(7) - boundCond.col(3) / 1.5) * t3 +
                    (6.5 * boundCond.col(6) - 7.5 * boundCond.col(2)) * t2 +
                    (-34.0 * boundCond.col(5) - 36.0 * boundCond.col(1)) * t1 +
                    (70.0 * boundCond.col(4) - 70.0 * boundCond.col(0));
  coeffMat.col(2) = (0.5 * boundCond.col(7) + boundCond.col(3)) * t3 +
                    (-7.0 * boundCond.col(6) + 10.0 * boundCond.col(2)) * t2 +
                    (39.0 * boundCond.col(5) + 45.0 * boundCond.col(1)) * t1 +
                    (-84.0 * boundCond.col(4) + 84.0 * boundCond.col(0));
  coeffMat.col(3) = (-boundCond.col(7) / 6.0 - boundCond.col(3) / 1.5) * t3 +
                    (2.5 * boundCond.col(6) - 5.0 * boundCond.col(2)) * t2 +
                    (-15.0 * boundCond.col(5) - 20.0 * boundCond.col(1)) * t1 +
                    (35.0 * boundCond.col(4) - 35.0 * boundCond.col(0));
  coeffMat.col(4) = boundCond.col(3) / 6.0;
  coeffMat.col(5) = boundCond.col(2) / 2.0;
  coeffMat.col(6) = boundCond.col(1);
  coeffMat.col(7) = boundCond.col(0);

  coeffMat.col(0) = coeffMat.col(0) / t7;
  coeffMat.col(1) = coeffMat.col(1) / t6;
  coeffMat.col(2) = coeffMat.col(2) / t5;
  coeffMat.col(3) = coeffMat.col(3) / t4;
}


bool TrajOpt::generate_traj(const Eigen::MatrixXd& iniState,
                            const Eigen::MatrixXd& init_yaw,
                            const Eigen::Vector3d& car_p,
                            const Eigen::Vector3d& car_v,
                            const double& car_theta,
                            const double& car_omega,
                            const Eigen::Quaterniond& land_q,
                            const int& N,
                            Trajectory<7>& traj,
                            const double& t_replan) {                 
  INFO_MSG("------seg: " << N);
  N_ = N;
  dim_t_ = 1;
  dim_p_ = N_ - 1;
  x_ = new double[dim_t_ + 4 * dim_p_ + 1 + 2 + 2];  // 1: tail thrust; ; 2: tail vt
  double& t = x_[0];
  Eigen::Map<Eigen::MatrixXd> P(x_ + dim_t_, 3, dim_p_);
  Eigen::Map<Eigen::MatrixXd> yaw(x_ + dim_t_ + dim_p_ * 3, 1, dim_p_);
  double& tail_f = x_[dim_t_ + 4 * dim_p_];
  Eigen::Map<Eigen::Vector2d> vt(x_ + dim_t_ + 4 * dim_p_ + 1);
  Eigen::Map<Eigen::Vector2d> at(x_ + dim_t_ + 4 * dim_p_ + 1 + 2);

  car_p_ = car_p;
  car_v_ = car_v;
  car_theta_ = car_theta;
  car_omega_ = car_omega;
  if (car_omega_ > 0.5){
    car_omega_ = 0.5;
  }
  v_max_horiz_ = car_v.head(2).norm() + dv_max_horiz_land_;
  // std::cout << "land_q: "
  //           << land_q.w() << ","
  //           << land_q.x() << ","
  //           << land_q.y() << ","
  //           << land_q.z() << "," << std::endl;
  q2v(land_q, tail_q_v_);
  INFO_MSG("tail_q_v_: " << tail_q_v_.transpose());
  theta_d_ = acos(tail_q_v_.dot(Eigen::Vector3d(0, 0, 1))); // acos return 0-pi
  theta_f_ = M_PI_2 - theta_d_;
  INFO_MSG("theta_d_: " << theta_d_ << ", theta_f_: " << theta_f_);
  assert(theta_d_ >= 0 && theta_f_ >= 0 && theta_d_ <= M_PI_2 && theta_f_ <= M_PI_2);
  thrust_middle_ = (thrust_max_ + thrust_min_) / 2;
  thrust_half_ = (thrust_max_ - thrust_min_) / 2;

  land_v_ = car_v - tail_q_v_ * v_plus_;
  std::cout << "tail_q_v_: " << tail_q_v_.transpose() << std::endl;

  v_t_x_ = tail_q_v_.cross(Eigen::Vector3d(0, 0, 1));
  if (v_t_x_.squaredNorm() == 0) {
    v_t_x_ = tail_q_v_.cross(Eigen::Vector3d(0, 1, 0));
  }
  v_t_x_.normalize();
  v_t_y_ = tail_q_v_.cross(v_t_x_);
  v_t_y_.normalize();

  vt.setConstant(0.0);
  at.setConstant(0.0);

  // NOTE set boundary conditions
  initS_ = iniState;
  init_yaw_.resize(1, 2);
  init_yaw_ = init_yaw;
  final_yaw_.resize(1, 2);
  final_yaw_.setZero();
  final_yaw_(0, 0) = rot_util::quaternion2yaw(land_q);
  // final_yaw_(0, 0) = 1.0;

  final_yaw_(0, 0) = init_yaw_(0, 0) + rot_util::error_angle(init_yaw_(0, 0), final_yaw_(0, 0));

  // set initial guess with obvp minimum jerk + rhoT
  minco_s4u_opt_.reset(N_);
  minco_s2_yaw_opt_.reset(init_yaw_, final_yaw_, N_);

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

  tail_f = 0;

  bool opt_once = initial_guess_ && t_replan > 0 && t_replan < init_traj_.getTotalDuration();
  short_mode_ = false;
  expect_traj_duration_ = 0.0;
  if (initial_guess_){
    expect_traj_duration_ = init_traj_.getTotalDuration() - t_replan;
    if (expect_traj_duration_ < short_mode_time_){
      INFO_MSG_YELLOW("short_mode!!!");
      with_perception_ = false;
      short_mode_ = true;
      if (!is_last_succ_){
        INFO_MSG_RED("[traj_opt] last opt fail, no reference! exit opt");
        delete[] x_;
        is_last_succ_ = false;
        return false;
      }
    }
    INFO_MSG_YELLOW("expect_traj_duration: " << expect_traj_duration_);
    set_rho_T_land(rhoT_land_origin_ + 
      rhoT_land_origin_/pow(std::max(expect_traj_duration_, 0.0) + 1.0, 3));
  }
  if (opt_once) {
    double init_T = init_traj_.getTotalDuration() - t_replan;
    t = logC2(init_T / N_);
    for (int i = 1; i < N_; ++i) {
      double tt0 = (i * 1.0 / N_) * init_T;
      P.col(i - 1) = init_traj_.getPos(tt0 + t_replan);
    }
    INFO_MSG("t: " << t);
    INFO_MSG("P: " << P.transpose());

    tail_f = init_tail_f_;
    vt = init_vt_;
    at = init_at_;
  } else {
    INFO_MSG_YELLOW("[traj_opt] Calculate BVP for initail");
    Eigen::MatrixXd bvp_i = initS_;
    Eigen::MatrixXd bvp_f(3, 4);
    Eigen::Vector3d car_p1, grad_carp_t;
    bvp_f.col(0) = car_p_;
    bvp_f.col(1) = car_v_;
    // bvp_f.col(2) = forward_thrust(tail_f) * tail_q_v_ + g_;
    bvp_f.col(2) = forward_thrust(tail_f) * Eigen::Vector3d(0, 0, 1) + g_;
    bvp_f.col(3).setZero();
    double T_bvp = (bvp_f.col(0) - bvp_i.col(0)).norm() / 0.5; // init v
    Piece<7>::CoefficientMat coeffMat;
    double max_omega = 0;
    do {
      T_bvp += 1.0;
      CYRV_model(car_p_, car_v_, car_theta_, car_omega_, T_bvp, car_p1, grad_carp_t);
      bvp_f.col(0) = car_p_ + car_v_ * T_bvp;
      bvp_f.col(0) = car_p1;
      bvp(T_bvp, bvp_i, bvp_f, coeffMat);
      std::vector<double> durs{T_bvp};
      std::vector<Piece<7>::CoefficientMat> coeffs{coeffMat};
      Trajectory<7> traj(durs, coeffs);
      max_omega = getMaxOmega(traj);
    } while (max_omega > 1.5 * omega_max_);

    Eigen::VectorXd tt(8);
    tt(7) = 1.0;
    for (int i = 1; i < N_; ++i) {
      double tt0 = (i * 1.0 / N_) * T_bvp;
      for (int j = 6; j >= 0; j -= 1) {
        tt(j) = tt(j + 1) * tt0;
      }
      P.col(i - 1) = coeffMat * tt;
    }
    t = logC2(T_bvp / N_);
  }
  std::cout << "initial guess >>> t: " << expC2(t)*N_ << std::endl;
  std::cout << "initial guess >>> tail_f: " << tail_f << std::endl;
  std::cout << "initial guess >>> vt: " << vt.transpose() << std::endl;

  // init yaw
  for (int i = 0; i < N_-1; ++i){
    // yaw(0, i) = init_yaw_(0, 0) + (final_yaw_(0, 0) - init_yaw_(0, 0)) * (i + 1) / N_;
    yaw(0, i) = final_yaw_(0, 0);
  }

  if (opt_once){
    double dT0 = expC2(t);
    minco_s4u_opt_.generate(initS_, tailS_, P, dT0);
    INFO_MSG_YELLOW("!!!!! init snap: " << minco_s4u_opt_.getTrajSnapCost());
    std::cout << "init tailS: " << tailS_.col(0).transpose() << std::endl;
  }



  // NOTE optimization
  lbfgs::lbfgs_parameter_t lbfgs_params;
  lbfgs::lbfgs_load_default_parameters(&lbfgs_params);
  lbfgs_params.mem_size = 32;
  lbfgs_params.past = 3;
  lbfgs_params.g_epsilon = 0.0;
  lbfgs_params.min_step = 1e-16;
  lbfgs_params.delta = 1e-4;
  lbfgs_params.line_search_type = 0;
  lbfgs_params.max_iterations = max_iter_;
  double minObjective;

  if (short_mode_){
    // lbfgs_params.past = 5;
    // lbfgs_params.delta = 1e-5;
  }

  int opt_ret = 0;

  auto tic = std::chrono::steady_clock::now();
  tictoc_innerloop_ = 0;
  tictoc_integral_ = 0;
  // while (cost_lock_.test_and_set())
  //   ;      
  iter_times_ = 0;
  opt_start_t_ = TimeNow();
  INFO_MSG_BLUE("with_perception: " << with_perception_);
  opt_ret = lbfgs::lbfgs_optimize(dim_t_ + 4 * dim_p_ + 1 + 2 + 2, x_, &minObjective,
                                  &objectiveFuncLanding, nullptr,
                                  &earlyExitLanding, this, &lbfgs_params);

  auto toc = std::chrono::steady_clock::now();
  // cost_lock_.clear();

  std::cout << "\033[32m>ret: " << opt_ret << "\033[0m" << std::endl;
  dashboard_cost_print();

  // std::cout << "innerloop costs: " << tictoc_innerloop_ * 1e-6 << "ms" << std::endl;
  // std::cout << "integral costs: " << tictoc_integral_ * 1e-6 << "ms" << std::endl;
  std::cout << "optmization costs: " << (toc - tic).count() * 1e-6 << "ms" << std::endl;
  std::cout << "\033[32m>iter times: " << iter_times_ << "\033[0m" << std::endl;
  if (pause_debug_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }

  double dT = expC2(t);
  double T = N_ * dT;
  Eigen::Vector3d tailV;
  forwardTailV(vt, tailV);
  std::cout << "vt: " << vt.transpose() << std::endl;
  std::cout << "at: " << at.transpose() << std::endl;

  if (opt_ret < 0) {
    delete[] x_;
    is_last_succ_ = false;
    return false;
  }

  Eigen::MatrixXd tailS(3, 4);
  tailS.col(0) = car_p_ + car_v_ * T + tail_q_v_ * robot_l_;
  Eigen::Vector3d car_p1, grad_carp_t;
  CYRV_model(car_p_, car_v_, car_theta_, car_omega_, T, car_p1, grad_carp_t);
  tailS.col(0) = car_p1 + tail_q_v_ * robot_l_;
  tailS.col(1) = tailV; 
  tailS.col(2) = forward_thrust(tail_f) * tail_q_v_ + g_ + at.x() * v_t_x_ + at.y() * v_t_y_;
  tailS.col(3).setZero();
  // std::cout << "tail thrust: " << forward_thrust(tail_f) << std::endl;
  std::cout << "tailS: " << tailS.col(0).transpose() << std::endl;
  minco_s4u_opt_.generate(initS_, tailS, P, dT);
  Eigen::VectorXd T_vec;
  T_vec.resize(N_);
  T_vec.setConstant(dT);
  minco_s2_yaw_opt_.generate(yaw, T_vec);

  traj = getS4UTrajWithYaw(minco_s4u_opt_, minco_s2_yaw_opt_);

  double max_omega = getMaxOmega(traj);
  std::cout << "maxOmega: " << max_omega << std::endl;
  std::cout << "maxThrust: " << traj.getMaxThrust() << std::endl;

  if (max_omega > omega_max_){
    INFO_MSG_RED("[traj_opt] Omega Exceed: " << max_omega);
    delete[] x_;
    is_last_succ_ = false;
    return false;
  } 

  double max_vel = traj.getMaxVelRate();
  if (max_vel > 1.5*vmax_){
    INFO_MSG_RED("[traj_opt] vel Exceed: " << max_vel);
    delete[] x_;
    is_last_succ_ = false;
    return false;
  } 

  double total_time = traj.getTotalDuration();
  if (initial_guess_ && total_time > std::max(3.0*expect_traj_duration_, 2.5)){ // TODO magic number
    INFO_MSG_RED("[traj_opt] Time Exceed: " << total_time);
    delete[] x_;
    is_last_succ_ = false;
    return false;
  }

  tailS_ = tailS;
  init_traj_ = traj;
  init_tail_f_ = tail_f;
  init_vt_ = vt;
  init_at_ = at;
  initial_guess_ = true;
  delete[] x_;
  is_last_succ_ = true;
  return true;
}

inline int earlyExitLanding(void* ptrObj,
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
  if (obj.pause_debug_) { // (obj.short_mode_ && expect_traj_duration_ < 0.2) || 
    const double& t = x[0];
    Eigen::Map<const Eigen::MatrixXd> P(x + obj.dim_t_, 3, obj.dim_p_);
    Eigen::Map<const Eigen::MatrixXd> yaw(x + obj.dim_t_ + obj.dim_p_ * 3, 1, obj.dim_p_);
    const double& tail_f = x[obj.dim_t_ + obj.dim_p_ * 4];
    Eigen::Map<const Eigen::Vector2d> vt(x + obj.dim_t_ + 4 * obj.dim_p_ + 1);
    Eigen::Map<const Eigen::Vector2d> at(x + obj.dim_t_ + 4 * obj.dim_p_ + 1 + 2);

    double dT = expC2(t);
    double T = obj.N_ * dT;
    Eigen::VectorXd T_vec;
    T_vec.resize(obj.N_);
    T_vec.setConstant(dT);

    Eigen::Vector3d tailV;
    forwardTailV(vt, tailV);
    Eigen::MatrixXd tailS(3, 4);
    tailS.col(0) = car_p_ + car_v_ * T + tail_q_v_ * obj.robot_l_;
    Eigen::Vector3d car_p1, grad_carp_t;
    CYRV_model(car_p_, car_v_, car_theta_, car_omega_, T, car_p1, grad_carp_t);
    tailS.col(0) = car_p1 + tail_q_v_ * obj.robot_l_;
    tailS.col(1) = tailV;
    tailS.col(2) = forward_thrust(tail_f) * tail_q_v_ + g_ + at.x() * v_t_x_ + at.y() * v_t_y_;
    tailS.col(3).setZero();

  std::cout << "tailS: " << tailS.col(0).transpose() << std::endl;

    obj.minco_s4u_opt_.generate(obj.initS_, tailS, P, dT);
    obj.minco_s2_yaw_opt_.generate(yaw, T_vec);


    auto traj = getS4UTrajWithYaw(obj.minco_s4u_opt_, obj.minco_s2_yaw_opt_);
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

    INFO_MSG_YELLOW("in process");
    obj.dashboard_cost_print();
    std::cout << "vt: " << vt.transpose() << std::endl;
    std::cout << "at: " << at.transpose() << std::endl;

    int a;
    std::cin >> a;
    // NOTE pause
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  // return k > 1e3;
  return 0;
}

// SECTION object function
inline double objectiveFuncLanding(void* ptrObj,
                                   const double* x,
                                   double* grad,
                                   const int n) {
  // std::cout << "damn n:" << n << std::endl;
  TrajOpt& obj = *(TrajOpt*)ptrObj;
  obj.iter_times_++;
  if (obj.iter_times_ % 50 == 0){
    INFO_MSG_RED("[traj_opt] iter: " << obj.iter_times_<<", time: " << durationSecond(TimeNow(), obj.opt_start_t_));
  }
  obj.clear_cost_rec();

  //! 1. fetch opt varaibles from x_
  const double& t = x[0];
  double& gradt = grad[0];
  Eigen::Map<const Eigen::MatrixXd> P(x + obj.dim_t_, 3, obj.dim_p_);
  Eigen::Map<Eigen::MatrixXd> gradP(grad + obj.dim_t_, 3, obj.dim_p_);

  Eigen::Map<const Eigen::MatrixXd> yaw(x + obj.dim_t_ + obj.dim_p_ * 3, 1, obj.dim_p_);
  Eigen::Map<Eigen::MatrixXd> grad_yaw(grad + obj.dim_t_ + obj.dim_p_ * 3, 1, obj.dim_p_);

  const double& tail_f = x[obj.dim_t_ + obj.dim_p_ * 4];
  double& grad_f = grad[obj.dim_t_ + obj.dim_p_ * 4];

  Eigen::Map<const Eigen::Vector2d> vt(x + obj.dim_t_ + 4 * obj.dim_p_ + 1);
  Eigen::Map<Eigen::Vector2d> grad_vt(grad + obj.dim_t_ + 4 * obj.dim_p_ + 1);

  Eigen::Map<const Eigen::Vector2d> at(x + obj.dim_t_ + 4 * obj.dim_p_ + 1 + 2);
  Eigen::Map<Eigen::Vector2d> grad_at(grad + obj.dim_t_ + 4 * obj.dim_p_ + 1 + 2);

  //! 2. reform P & T
  double dT = expC2(t);
  Eigen::VectorXd T_vec;
  T_vec.resize(obj.N_);
  T_vec.setConstant(dT);

  //! 3. calculate ternimal state
  Eigen::Vector3d tailV, grad_tailV, grad_tailA;
  forwardTailV(vt, tailV);
  Eigen::MatrixXd tailS(3, 4);
  tailS.col(0) = car_p_ + car_v_ * obj.N_ * dT + tail_q_v_ * obj.robot_l_;
  Eigen::Vector3d car_p1, grad_carp_t;
  CYRV_model(car_p_, car_v_, car_theta_, car_omega_, obj.N_ * dT, car_p1, grad_carp_t);
  tailS.col(0) = car_p1 + tail_q_v_ * obj.robot_l_;
  tailS.col(1) = tailV; 
  tailS.col(2) = forward_thrust(tail_f) * tail_q_v_ + g_ + at.x() * v_t_x_ + at.y() * v_t_y_;
  tailS.col(3).setZero();

  //! 4. generate minco using P & T
  auto tic = std::chrono::steady_clock::now();
  obj.minco_s4u_opt_.generate(obj.initS_, tailS, P, dT);
  obj.minco_s2_yaw_opt_.generate(yaw, T_vec);

  double cost = obj.minco_s4u_opt_.getTrajSnapCost();
  obj.minco_s4u_opt_.calGrads_CT();
  obj.cost_snap_rec_ = cost;

  cost += obj.minco_s2_yaw_opt_.getEnergyCost();
  obj.minco_s2_yaw_opt_.calGrads_CT();

  auto toc = std::chrono::steady_clock::now();
  tictoc_innerloop_ += (toc - tic).count();
  // double cost_with_only_energy = cost;
  // std::cout << "cost of energy: " << cost_with_only_energy << std::endl;

  //! 5. calculate penalty and gradient to C & T
  tic = std::chrono::steady_clock::now();
  obj.addTimeIntPenaltyLanding(cost);
  toc = std::chrono::steady_clock::now();
  tictoc_integral_ += (toc - tic).count();

  //! 6. propogate gradient to mid-point P & T
  tic = std::chrono::steady_clock::now();
  obj.minco_s4u_opt_.calGrads_PT();
  obj.minco_s2_yaw_opt_.calGrads_PT();
  toc = std::chrono::steady_clock::now();
  tictoc_innerloop_ += (toc - tic).count();
  // std::cout << "cost of penalty: " << cost - cost_with_only_energy << std::endl;

  obj.minco_s4u_opt_.gdT += obj.minco_s4u_opt_.gdTail.col(0).dot(obj.N_ * grad_carp_t);
  // obj.minco_s4u_opt_.gdT += obj.minco_s4u_opt_.gdTail.col(0).dot(obj.N_ * car_v_);
  grad_tailV = obj.minco_s4u_opt_.gdTail.col(1);
  grad_tailA = obj.minco_s4u_opt_.gdTail.col(2);

  double grad_thrust = obj.minco_s4u_opt_.gdTail.col(2).dot(tail_q_v_);
  addLayerThrust(tail_f, grad_thrust, grad_f);

  grad_vt.setZero();
  if (obj.rhoVt_ > -1) {
    grad_vt.x() = grad_tailV.dot(v_t_x_);
    grad_vt.y() = grad_tailV.dot(v_t_y_);
    double vt_sqr = vt.squaredNorm();
    cost += obj.rhoVt_ * vt_sqr;
    grad_vt += obj.rhoVt_ * 2 * vt;
  }

  grad_at.setZero();
  if (obj.rhoAt_ > -1) {
    grad_at.x() = grad_tailA.dot(v_t_x_);
    grad_at.y() = grad_tailA.dot(v_t_y_);
    double at_sqr = at.squaredNorm();
    cost += obj.rhoAt_ * at_sqr;
    grad_at += obj.rhoAt_ * 2 * at;
  }

  //! 7. propogate gradient to opt variable P & tau
  gradt = 0.0;
  obj.minco_s4u_opt_.gdT += obj.rhoT_land_;
  if (!obj.short_mode_){
    cost += obj.rhoT_land_ * dT;
    gradt = (obj.minco_s4u_opt_.gdT + obj.minco_s2_yaw_opt_.gdT.sum()) * gdT2t(t);
  }
  obj.cost_t_rec_ = obj.rhoT_land_ * dT;

  gradP = obj.minco_s4u_opt_.gdP;
  grad_yaw = obj.minco_s2_yaw_opt_.gdP;

  return cost;
}
// !SECTION object function


void TrajOpt::addTimeIntPenaltyLanding(double& cost) {
  // cost of one inner sample points
  double cost_inner;
  // state of innner sample point 
  Eigen::Vector3d pos, vel, acc, jer, snp;
  double psi, dPsi, ddPsi, thr;
  thr = 0.0;
  Eigen::Vector4d quat;
  Eigen::Vector3d bodyrate;
  // gradient of the state of innner sample point 
  Eigen::Vector3d grad_p, grad_v, grad_a, grad_j, grad_omega;
  double grad_thr;
  Eigen::Vector4d grad_quat;
  // total gradient
  Eigen::Vector3d totalGradPos, totalGradVel, totalGradAcc, totalGradJer;
  double totalGradPsi = 0.0, totalGradPsiD = 0.0;
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
  double cost_vel, cost_thrust, cost_tmp, cost_rate, cost_vel_yaw;
  Eigen::Vector3d grad_vel, grad_thrust, grad_tmp, grad_tmp2, grad_tmp3, grad_rate;
  double grad_vel_yaw;
  cost_rate = 0.0;
  
  int innerLoop = K_ + 1;
  step = minco_s4u_opt_.t(1) / K_;

  for (int j = 0; j < innerLoop; ++j) {
    beta0 = cal_timebase_snap(0, st);
    beta1 = cal_timebase_snap(1, st);
    beta2 = cal_timebase_snap(2, st);
    beta3 = cal_timebase_snap(3, st);
    beta4 = cal_timebase_snap(4, st);
    beta_yaw0 = cal_timebase_acc(0, st);
    beta_yaw1 = cal_timebase_acc(1, st);
    beta_yaw2 = cal_timebase_acc(2, st);
    alpha = 1.0 / K_ * j;
    omg = (j == 0 || j == innerLoop - 1) ? 0.5 : 1.0;

    for (int i = 0; i < N_; ++i) {
      const auto& c = minco_s4u_opt_.c.block<8, 3>(i * 8, 0);
      const auto& c_yaw = minco_s2_yaw_opt_.b.block<4, 1>(i * 4, 0);

      pos = c.transpose() * beta0;
      vel = c.transpose() * beta1;
      acc = c.transpose() * beta2;
      jer = c.transpose() * beta3;
      snp = c.transpose() * beta4;
      psi   = c_yaw.transpose() * beta_yaw0;
      dPsi  = c_yaw.transpose() * beta_yaw1;
      ddPsi = c_yaw.transpose() * beta_yaw2;
      // flatmap_.forward(vel, acc, jer, psi, dPsi, thr, quat, bodyrate);

      grad_p.setZero();
      grad_v.setZero();
      grad_a.setZero();
      grad_j.setZero();
      grad_omega.setZero();
      grad_thr = 0; //no use
      grad_quat.setZero(); //no use
      cost_inner = 0.0;

      if (grad_cost_floor(pos, grad_tmp, cost_tmp)) {
        grad_p += grad_tmp;
        cost_inner += cost_tmp;
      }

      if (grad_cost_v(vel, v_max_horiz_, v_max_vert_land_, grad_vel, cost_vel)) {
        grad_v += grad_vel;
        cost_inner += cost_vel;
        cost_v_rec_ += omg * step * cost_vel;
      }

      if (grad_cost_thrust(acc, grad_thrust, cost_thrust)) {
        grad_a += grad_thrust;
        cost_inner += cost_thrust;
        cost_thrust_rec_ += omg * step * cost_thrust;
      }

      // if (grad_cost_rate(bodyrate, grad_rate, cost_rate)){
      //   grad_omega += grad_rate;
      //   cost_inner += cost_rate;
      //   cost_omega_rec_ += omg * step * cost_rate;
      // }

      if (grad_cost_omega(acc, jer, grad_tmp, grad_tmp2, cost_tmp)) {
        grad_a += grad_tmp;
        grad_j += grad_tmp2;
        cost_inner += cost_tmp;
        cost_omega_rec_ += omg * step * cost_tmp;
      }

      if (grad_cost_dyaw(dPsi, grad_vel_yaw, cost_vel_yaw)) {
        totalGradPsiD += grad_vel_yaw;
        cost_inner += cost_vel_yaw;
        cost_dyaw_rec_ += omg * step * cost_vel_yaw;
      }

      // if (grad_cost_omega_yaw(acc, jer, grad_tmp, grad_tmp2, cost_tmp)) {
      //   grad_a += grad_tmp;
      //   grad_j += grad_tmp2;
      //   cost_inner += cost_tmp;
      // }

      double dur2now = (i + alpha) * minco_s4u_opt_.t(1);
      // Eigen::Vector3d car_p = car_p_ + car_v_ * dur2now;
      Eigen::Vector3d car_p, grad_carp_t;
      CYRV_model(car_p_, car_v_, car_theta_, car_omega_, dur2now, car_p, grad_carp_t);

      if (grad_cost_perching_collision(pos, acc, car_p,
                                       grad_tmp, grad_tmp2, grad_tmp3,
                                       cost_tmp)) {
        grad_p += grad_tmp;
        grad_a += grad_tmp2;
        cost_inner += cost_tmp;
        cost_perching_collision_rec_ += omg * step * cost_tmp;
      }
      // double grad_car_t = grad_tmp3.dot(car_v_);
      double grad_car_t = grad_tmp3.dot(grad_carp_t);

      if (with_perception_){
        if (grad_cost_preception(pos, acc, car_p,
                                grad_tmp, grad_tmp2, grad_tmp3,
                                cost_tmp)) {
          grad_p += grad_tmp;
          grad_a += grad_tmp2;
          cost_inner += cost_tmp;
          // INFO_MSG("grad_cost_preception: " << cost_tmp);
          cost_perching_precep_rec_ += omg * step * cost_tmp;
        }
        // grad_car_t += grad_tmp3.dot(car_v_);
        grad_car_t += grad_tmp3.dot(grad_carp_t);
      }


      // flatmap_.backward(grad_p, grad_v, grad_a, grad_thr, grad_quat, grad_omega,
      //                                totalGradPos, totalGradVel, totalGradAcc, totalGradJer,
      //                                totalGradPsi, totalGradPsiD);


      // gradViola_c = beta0 * totalGradPos.transpose();
      // gradViola_t = totalGradPos.transpose() * vel;
      // gradViola_c += beta1 * totalGradVel.transpose();
      // gradViola_t += totalGradVel.transpose() * acc;
      // gradViola_c += beta2 * totalGradAcc.transpose();
      // gradViola_t += totalGradAcc.transpose() * jer;
      // gradViola_c += beta3 * totalGradJer.transpose();
      // gradViola_t += totalGradJer.transpose() * snp;
      // gradViola_t += grad_car_t;

      gradViola_c = beta0 * grad_p.transpose();
      gradViola_t = grad_p.transpose() * vel;
      gradViola_c += beta1 * grad_v.transpose();
      gradViola_t += grad_v.transpose() * acc;
      gradViola_c += beta2 * grad_a.transpose();
      gradViola_t += grad_a.transpose() * jer;
      gradViola_c += beta3 * grad_j.transpose();
      gradViola_t += grad_j.transpose() * snp;
      gradViola_t += grad_car_t;

      gradViola_c_yaw = beta_yaw0 * totalGradPsi;
      gradViola_t_yaw = totalGradPsi * dPsi;
      gradViola_c_yaw += beta_yaw1 * totalGradPsiD;
      gradViola_t_yaw += totalGradPsiD * ddPsi;

      minco_s4u_opt_.gdC.block<8, 3>(i * 8, 0) += omg * step * gradViola_c;
      minco_s4u_opt_.gdT += omg * (cost_inner / K_ + alpha * step * gradViola_t + alpha * step * gradViola_t_yaw);
      minco_s4u_opt_.gdT += i * omg * step * grad_car_t;

      minco_s2_yaw_opt_.gdC.block<4, 1>(i * 4, 0) += omg * step * gradViola_c_yaw;

      cost += omg * step * cost_inner;
    }
    st += step;
  }
}

bool TrajOpt::grad_cost_thrust(const Eigen::Vector3d& a,
                               Eigen::Vector3d& grada,
                               double& costa) {
  bool ret = false;
  grada.setZero();
  costa = 0;
  Eigen::Vector3d thrust_f = a - g_;
  double max_pen = thrust_f.squaredNorm() - thrust_max_ * thrust_max_;
  if (max_pen > 0) {
    double grad = 0;
    costa = rhoThrust_ * smoothedL1(max_pen, grad);
    grada = rhoThrust_ * 2 * grad * thrust_f;
    ret = true;
  }

  double min_pen = thrust_min_ * thrust_min_ - thrust_f.squaredNorm();
  if (min_pen > 0) {
    double grad = 0;
    costa = rhoThrust_ * smoothedL1(min_pen, grad);
    grada = -rhoThrust_ * 2 * grad * thrust_f;
    ret = true;
  }

  return ret;
}

// using hopf fibration:
// [a,b,c] = thrust.normalized()
// \omega_1 = sin(\phi) \dot{a] - cos(\phi) \dot{b} - (a sin(\phi) - b cos(\phi)) (\dot{c}/(1+c))
// \omega_2 = cos(\phi) \dot{a] - sin(\phi) \dot{b} - (a cos(\phi) - b sin(\phi)) (\dot{c}/(1+c))
// \omega_3 = (b \dot{a} - a \dot(b)) / (1+c)
// || \omega_12 ||^2 = \omega_1^2 + \omega_2^2 = \dot{a}^2 + \dot{b}^2 + \dot{c}^2

bool TrajOpt::grad_cost_omega(const Eigen::Vector3d& a,
                              const Eigen::Vector3d& j,
                              Eigen::Vector3d& grada,
                              Eigen::Vector3d& gradj,
                              double& cost) {
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

bool TrajOpt::grad_cost_rate(const Eigen::Vector3d& omg,
                                Eigen::Vector3d& gradomg,
                                double& cost)
{
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
    for(int i = 0; i < 3; i++)
    {
        double pen = rate_vec(i) * rate_vec(i) - max_vec(i) * max_vec(i);
        if( pen > 0.0)
        {
            double pen2 = pen * pen;
            cost += rhoOmega_ * pen *  pen2;
            gradomg += rhoOmega_ * 3 * pen2 * gradtmp.block<1, 3>(i, 0).transpose();
        }
    }
    return true;
}

bool TrajOpt::grad_cost_omega_yaw(const Eigen::Vector3d& a,
                                  const Eigen::Vector3d& j,
                                  Eigen::Vector3d& grada,
                                  Eigen::Vector3d& gradj,
                                  double& cost) {
  // TODO
  return false;
}

bool TrajOpt::grad_cost_floor(const Eigen::Vector3d& p,
                              Eigen::Vector3d& gradp,
                              double& costp) {
  costp = 0.0;
  gradp.setZero();
  static double z_floor = land_z_min_;
  double pen = z_floor - p.z();
  if (pen > 0) {
    double grad = 0;
    costp = smoothedL1(pen, grad);
    costp *= rhoP_;
    gradp.setZero();
    gradp.z() = -rhoP_ * grad;
    return true;
  } else {
    return false;
  }
}

bool TrajOpt::grad_cost_preception(const Eigen::Vector3d& pos,
                                   const Eigen::Vector3d& acc,
                                   const Eigen::Vector3d& car_p,
                                   Eigen::Vector3d& gradp,
                                   Eigen::Vector3d& grada,
                                   Eigen::Vector3d& grad_car_p,
                                   double& cost) 
{
  static double eps = 1e-6;
  static double eps_pz = 0.1;
  gradp.setZero();
  grada.setZero();
  grad_car_p.setZero();
  cost = 0.0;
  static bool is_use_grada = false; 

  double dist_sqr = (pos - car_p).squaredNorm();
  double pen_dist1 = preception_d_max_ * preception_d_max_ - dist_sqr;
  double pen_dist2 = dist_sqr - preception_d_min_ * preception_d_min_;

  double grad_dist1 = 0, grad_dist2 = 0;
  double F2_1 = smoothed01(pen_dist1, grad_dist1, 0.01);
  double F2_2 = smoothed01(pen_dist2, grad_dist2, 0.01);
  double F2 =  F2_1 * F2_2;

  // INFO_MSG_BLUE("pos: " << pos.transpose() <<", pen_dist: "<<pen_dist);

  Eigen::Vector3d grad_F2_pos = -2.0 * (pos - car_p) * grad_dist1 * F2_2 + 2.0 * (pos - car_p) * grad_dist2 * F2_1; 
  Eigen::Vector3d grad_F2_carp = -grad_F2_pos;

  Eigen::Quaterniond qbw;
  Eigen::Vector3d thrust_f;
  double a = 0.0;
  double b = 0.0;
  double c = 0.0;
  if (is_use_grada){
    thrust_f = acc - g_;
    Eigen::Vector3d zb = f_N(thrust_f);

    a = zb.x();
    b = zb.y();
    c = zb.z();

    Eigen::Quaterniond qwb;
    qwb.w() = sqrt(1+c) / sqrt(2.0);
    qwb.x() = -b / (sqrt(2.0*(1+c)) + eps);
    qwb.y() =  a / (sqrt(2.0*(1+c)) + eps);
    qwb.z() = 0.0;

    qbw.w() = qwb.w(); qbw.x() = -qwb.x(); qbw.y() = -qwb.y(); qbw.z() = -qwb.z();
  }else{
    qbw = Eigen::Quaterniond(1,0,0,0);
  }

  // double rho_front = 1; // cos(theta_f_) + 0.001;
  // double rho_down  = 1; //cos(theta_d_) + 0.001;

  double rho_front = cos(theta_f_) + 1;
  double rho_down  = cos(theta_d_) + 1;

  // For cost
  const Eigen::Quaterniond qcb_f(cam2body_R_front_.transpose());
  const Eigen::Vector3d pbc_f = cam2body_p_front_;
  Eigen::Matrix3d Rbc_f = cam2body_R_front_;

  Eigen::Vector3d pc_f; // target in camera frame
  pc_f = qcb_f * (qbw * (car_p - pos) - pbc_f); 
  // double u = pc.x() / (pc.z() + eps) * fx_ + cx_;
  // double v = pc.y() / (pc.z() + eps) * fy_ + cy_;
  // desire the target is in the middgle of camera frame
  pc_f.z() += eps_pz;
  double uc_f = pc_f.x() / (pc_f.z()) * fx_front_;
  double vc_f = pc_f.y() / (pc_f.z()) * fy_front_;
  double F_f = (uc_f * uc_f + vc_f * vc_f) * rho_front;

  const Eigen::Quaterniond qcb_d(cam2body_R_down_.transpose());
  const Eigen::Vector3d pbc_d = cam2body_p_down_;
  Eigen::Matrix3d Rbc_d = cam2body_R_down_;

  Eigen::Vector3d pc_d; // target in camera frame
  pc_d = qcb_d * (qbw * (car_p - pos) - pbc_d); 
  // double u = pc.x() / (pc.z() + eps) * fx_ + cx_;
  // double v = pc.y() / (pc.z() + eps) * fy_ + cy_;
  // desire the target is in the middgle of camera frame
  pc_d.z() += eps_pz;
  double uc_d = pc_d.x() / (pc_d.z()) * fx_down_;
  double vc_d = pc_d.y() / (pc_d.z()) * fy_down_;
  double F_d = (uc_d * uc_d + vc_d * vc_d) * rho_down;

  double grad_S_Ff, grad_S_Fd;
  double S = softmin(F_f, F_d, grad_S_Ff, grad_S_Fd);
  cost = S * F2;
  // cost = F_f * F2;
  INFO_MSG_BLUE("F_f: " << F_f <<", F_d: " << F_d);

  INFO_MSG_BLUE("S: " << S <<", F2: " << F2);

  // For par_Fd / par_pcd
  double grad_Fd_ud, grad_Fd_vd; 
  grad_Fd_ud = 2 * uc_d * rho_down;
  grad_Fd_vd = 2 * vc_d * rho_down;
  Eigen::Vector3d grad_Fd_pcd;
  Eigen::Vector3d grad_ud_pcd, grad_vd_pcd;
  grad_ud_pcd << fx_down_ / pc_d.z(), 0, -pc_d.x() / (pc_d.z() * pc_d.z()) * fx_down_;
  grad_vd_pcd << 0, fy_down_ / pc_d.z(), -pc_d.y() / (pc_d.z() * pc_d.z()) * fy_down_;
  grad_Fd_pcd = grad_ud_pcd * grad_Fd_ud + grad_vd_pcd * grad_Fd_vd;

  // For par_Ff / par_pcf
  double grad_Ff_uf, grad_Ff_vf; 
  grad_Ff_uf = 2 * uc_f * rho_front;
  grad_Ff_vf = 2 * vc_f * rho_front;
  Eigen::Vector3d grad_Ff_pcf;
  Eigen::Vector3d grad_uf_pcf, grad_vf_pcf;
  grad_uf_pcf << fx_front_ / pc_f.z(), 0, -pc_f.x() / (pc_f.z() * pc_f.z()) * fx_front_;
  grad_vf_pcf << 0, fy_front_ / pc_f.z(), -pc_f.y() / (pc_f.z() * pc_f.z()) * fy_front_;
  grad_Ff_pcf = grad_uf_pcf * grad_Ff_uf + grad_vf_pcf * grad_Ff_vf;

  // For par_cost / par_pos
  Eigen::Matrix3d grad_pcd_pos;
  grad_pcd_pos = -(qcb_d * qbw).toRotationMatrix().transpose();
  Eigen::Matrix3d grad_pcf_pos;
  grad_pcf_pos = -(qcb_f * qbw).toRotationMatrix().transpose();
  Eigen::Vector3d grad_Fd_pos = grad_pcd_pos * grad_Fd_pcd;
  Eigen::Vector3d grad_Ff_pos = grad_pcf_pos * grad_Ff_pcf;
  Eigen::Vector3d grad_S_pos = grad_Fd_pos * grad_S_Fd + grad_Ff_pos * grad_S_Ff;
  gradp = grad_S_pos * F2 + S * grad_F2_pos;
  // gradp = grad_Ff_pos * F2 + F_f * grad_F2_pos;


  // For par_cost / par_carp
  Eigen::Matrix3d grad_pcd_carp;
  grad_pcd_carp = (qcb_d * qbw).toRotationMatrix().transpose();
  Eigen::Matrix3d grad_pcf_carp;
  grad_pcf_carp = (qcb_d * qbw).toRotationMatrix().transpose();
  Eigen::Vector3d grad_Fd_carp = grad_pcd_carp * grad_Fd_pcd;
  Eigen::Vector3d grad_Ff_carp = grad_pcf_carp * grad_Ff_pcf;
  Eigen::Vector3d grad_S_carp = grad_Fd_carp * grad_S_Fd + grad_Ff_carp * grad_S_Ff;
  grad_car_p = grad_S_carp * F2 + S * grad_F2_carp;
  // grad_car_p = grad_Ff_carp * F2 + F_f * grad_F2_carp;


  // For par_cost / par_acc
  if (is_use_grada){
    Eigen::MatrixXd grad_pb_qbw;
    grad_pb_qbw.resize(4, 3);
    getJacobian(car_p - pos, qbw, grad_pb_qbw);
    Eigen::MatrixXd grad_pc_qbw;
    grad_pc_qbw.resize(4, 3);
    grad_pc_qbw = grad_pb_qbw * Rbc_d;

    Eigen::Vector4d grad_cost_qbw;
    grad_cost_qbw = grad_pc_qbw * grad_Fd_pcd;

    Eigen::Vector4d grad_cost_qwb;
    grad_cost_qwb = -grad_cost_qbw;
    grad_cost_qwb(0) = -grad_cost_qwb(0);

    Eigen::MatrixXd grad_qwb_zb;
    grad_qwb_zb.setZero(3, 4);
    grad_qwb_zb(0, 2) =   1 / (sqrt(2.0*(1+c)) + eps);
    grad_qwb_zb(1, 1) =  -1 / (sqrt(2.0*(1+c)) + eps);
    grad_qwb_zb(2, 0) =  1 / (2.0*sqrt(2.0)) * 1.0 / (sqrt(1+c) + eps);
    grad_qwb_zb(2, 1) =  b / (2.0*sqrt(2.0)) * 1.0 / (pow(sqrt(1+c), 3) + eps);
    grad_qwb_zb(2, 2) = -a / (2.0*sqrt(2.0)) * 1.0 / (pow(sqrt(1+c), 3) + eps);

    grada = F2 * f_DN(thrust_f).transpose() * grad_qwb_zb * grad_cost_qwb;
  }


  cost *= rhoPerchingPreception_;
  gradp *= rhoPerchingPreception_;
  grad_car_p *= rhoPerchingPreception_;
  grada *= rhoPerchingPreception_;

  return true;
}

// plate: \Epsilon = \left{ x = RBu + c | \norm(u) \leq r \right}
// x \in R_{3\times1}, u \in R_{2\times1}, B \in R_{3\times2}
// c: center of the plate; p: center of the drone bottom
//  c = p - l * z_b
// plane: a^T x \leq b
//        a^T(RBu + c) \leq b
//        a^T(RBu + p - l * z_b) \leq b
//        u^T(B^T R^T a) + a^Tp - a^T*l*z_b - b \leq 0
//        r \norm(B^T R^T a) + a^Tp - a^T*l*z_b - b \leq 0
// B^T R^T = [1-2y^2,    2xy, -2yw;
//               2xy, 1-2x^2,  2xw]
// B^T R^T = [1-a^2/(1+c),   -ab/(1+c), -a;
//              -ab/(1+c), 1-b^2/(1+c), -b]
bool TrajOpt::grad_cost_perching_collision(const Eigen::Vector3d& pos,
                                           const Eigen::Vector3d& acc,
                                           const Eigen::Vector3d& car_p,
                                           Eigen::Vector3d& gradp,
                                           Eigen::Vector3d& grada,
                                           Eigen::Vector3d& grad_car_p,
                                           double& cost) {
  static double eps = 1e-6;
  gradp.setZero();
  grada.setZero();
  grad_car_p.setZero();
  cost = 0.0;

  double dist_sqr = (pos - car_p).squaredNorm();
  double safe_r = platform_r_ + robot_r_;
  double safe_r_sqr = safe_r * safe_r;
  double pen_dist = safe_r_sqr - dist_sqr;
  pen_dist /= safe_r_sqr;
  double grad_dist = 0;
  double var01 = smoothed01(pen_dist, grad_dist);
  if (var01 == 0) {
    return false;
  }
  Eigen::Vector3d gradp_dist = grad_dist * 2 * (car_p - pos);
  Eigen::Vector3d grad_carp_dist = -gradp_dist;

  Eigen::Vector3d a_i = -tail_q_v_;
  double b_i = a_i.dot(car_p);

  Eigen::Vector3d thrust_f = acc - g_;
  Eigen::Vector3d zb = f_N(thrust_f);

  Eigen::MatrixXd BTRT(2, 3);
  double a = zb.x();
  double b = zb.y();
  double c = zb.z();

  double c_1 = 1.0 / (1 + c);

  BTRT(0, 0) = 1 - a * a * c_1;
  BTRT(0, 1) = -a * b * c_1;
  BTRT(0, 2) = -a;
  BTRT(1, 0) = -a * b * c_1;
  BTRT(1, 1) = 1 - b * b * c_1;
  BTRT(1, 2) = -b;

  Eigen::Vector2d v2 = BTRT * a_i;
  double v2_norm = sqrt(v2.squaredNorm() + eps);
  double pen = a_i.dot(pos) - (robot_l_ - 0.005) * a_i.dot(zb) - b_i + robot_r_ * v2_norm;

  if (pen > 0) {
    double grad = 0;
    cost = smoothedL1(pen, grad);
    // gradients: pos, car_p, v2
    gradp = a_i;
    grad_car_p = -a_i;
    Eigen::Vector2d grad_v2 = robot_r_ * v2 / v2_norm;

    Eigen::MatrixXd pM_pa(2, 3), pM_pb(2, 3), pM_pc(2, 3);
    double c2_1 = c_1 * c_1;

    pM_pa(0, 0) = -2 * a * c_1;
    pM_pa(0, 1) = -b * c_1;
    pM_pa(0, 2) = -1;
    pM_pa(1, 0) = -b * c_1;
    pM_pa(1, 1) = 0;
    pM_pa(1, 2) = 0;

    pM_pb(0, 0) = 0;
    pM_pb(0, 1) = -a * c_1;
    pM_pb(0, 2) = 0;
    pM_pb(1, 0) = -a * c_1;
    pM_pb(1, 1) = -2 * b * c_1;
    pM_pb(1, 2) = -1;

    pM_pc(0, 0) = a * a * c2_1;
    pM_pc(0, 1) = a * b * c2_1;
    pM_pc(0, 2) = 0;
    pM_pc(1, 0) = a * b * c2_1;
    pM_pc(1, 1) = b * b * c2_1;
    pM_pc(1, 2) = 0;

    Eigen::MatrixXd pv2_pzb(2, 3);
    pv2_pzb.col(0) = pM_pa * a_i;
    pv2_pzb.col(1) = pM_pb * a_i;
    pv2_pzb.col(2) = pM_pc * a_i;

    Eigen::Vector3d grad_zb = pv2_pzb.transpose() * grad_v2 - robot_l_ * a_i;

    grada = f_DN(thrust_f).transpose() * grad_zb;

    grad *= var01;
    gradp_dist *= cost;
    grad_carp_dist *= cost;
    cost *= var01;
    gradp = grad * gradp + gradp_dist;
    grada *= grad;
    grad_car_p = grad * grad_car_p + grad_carp_dist;

    cost *= rhoPerchingCollision_;
    gradp *= rhoPerchingCollision_;
    grada *= rhoPerchingCollision_;
    grad_car_p *= rhoPerchingCollision_;

    // std::cout << "var01: " << var01 << std::endl;

    return true;
  }
  return false;
}

bool TrajOpt::check_collilsion(const Eigen::Vector3d& pos,
                               const Eigen::Vector3d& acc,
                               const Eigen::Vector3d& car_p) {
  if ((pos - car_p).norm() > platform_r_) {
    return false;
  }
  static double eps = 1e-6;

  Eigen::Vector3d a_i = -tail_q_v_;
  double b_i = a_i.dot(car_p);

  Eigen::Vector3d thrust_f = acc - g_;
  Eigen::Vector3d zb = f_N(thrust_f);

  Eigen::MatrixXd BTRT(2, 3);
  double a = zb.x();
  double b = zb.y();
  double c = zb.z();

  double c_1 = 1.0 / (1 + c);

  BTRT(0, 0) = 1 - a * a * c_1;
  BTRT(0, 1) = -a * b * c_1;
  BTRT(0, 2) = -a;
  BTRT(1, 0) = -a * b * c_1;
  BTRT(1, 1) = 1 - b * b * c_1;
  BTRT(1, 2) = -b;

  Eigen::Vector2d v2 = BTRT * a_i;
  double v2_norm = sqrt(v2.squaredNorm() + eps);
  double pen = a_i.dot(pos) - (robot_l_ - 0.005) * a_i.dot(zb) - b_i + robot_r_ * v2_norm;

  return pen > 0;
}

bool TrajOpt::feasibleCheck(Trajectory<7>& traj) {
  double dt = 0.01;
  for (double t = 0; t < traj.getTotalDuration(); t += dt) {
    Eigen::Vector3d p = traj.getPos(t);
    Eigen::Vector3d a = traj.getAcc(t);
    Eigen::Vector3d j = traj.getJer(t);
    Eigen::Vector3d thrust = a - g_;
    Eigen::Vector3d zb_dot = f_DN(thrust) * j;
    double omega12 = zb_dot.norm();
    if (omega12 > omega_max_ + 0.2) {
      return false;
    }
    if (p.z() < 0.1) {
      return false;
    }
  }
  return true;
}

}  // namespace traj_opt