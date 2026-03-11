#include <traj_opt/traj_opt.h>

#include <traj_opt/lbfgs_raw.hpp>

namespace traj_opt {
using rot_util = rotation_util::RotUtil;

static Eigen::Vector3d car_p_, car_v_;
static Eigen::Vector3d tail_q_v_;
static Eigen::Vector3d g_(0, 0, -9.8);
static Eigen::Vector3d land_v_;
static Trajectory<7> init_traj_;
static double init_tail_f_;
static bool initial_guess_ = false;

static double thrust_middle_, thrust_half_;

static double tictoc_innerloop_;
static double tictoc_integral_;


static bool q2v(const Eigen::Quaterniond& q,
                Eigen::Vector3d& v) {
  Eigen::MatrixXd R = q.toRotationMatrix();
  v = R.col(2);
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
                            const Eigen::Vector3d& car_p,
                            const Eigen::Vector3d& car_v,
                            const Eigen::Quaterniond& land_q,
                            const int& N,
                            Trajectory<7>& traj,
                            const double& t_replan) {                 
  N_ = N;
  dim_t_ = 1;
  dim_p_ = N_ - 1;
  x_ = new double[dim_t_ + 3 * dim_p_ + 1];  // 1: tail thrust;
  double& t = x_[0];
  Eigen::Map<Eigen::MatrixXd> P(x_ + dim_t_, 3, dim_p_);
  double& tail_f = x_[dim_t_ + 3 * dim_p_];

  car_p_ = car_p;
  car_v_ = car_v;
  // std::cout << "land_q: "
  //           << land_q.w() << ","
  //           << land_q.x() << ","
  //           << land_q.y() << ","
  //           << land_q.z() << "," << std::endl;
  q2v(land_q, tail_q_v_);
  thrust_middle_ = (thrust_max_ + thrust_min_) / 2;
  thrust_half_ = (thrust_max_ - thrust_min_) / 2;

  land_v_ = car_v - tail_q_v_ * v_plus_;
  std::cout << "tail_q_v_: " << tail_q_v_.transpose() << std::endl;

  // NOTE set boundary conditions
  initS_ = iniState;

  // set initial guess with obvp minimum jerk + rhoT
  minco_s4u_opt_.reset(N_);

  tail_f = 0;

  bool opt_once = initial_guess_ && t_replan > 0 && t_replan < init_traj_.getTotalDuration();
  if (opt_once) {
    double init_T = init_traj_.getTotalDuration() - t_replan;
    t = logC2(init_T / N_);
    for (int i = 1; i < N_; ++i) {
      double tt0 = (i * 1.0 / N_) * init_T;
      P.col(i - 1) = init_traj_.getPos(tt0 + t_replan);
    }
    tail_f = init_tail_f_;
  } else {
    std::cout << "111"  << std::endl;

    Eigen::MatrixXd bvp_i = initS_;
    Eigen::MatrixXd bvp_f(3, 4);
    bvp_f.col(0) = car_p_;
    bvp_f.col(1) = car_v_;
    bvp_f.col(2) = forward_thrust(tail_f) * tail_q_v_ + g_;
    bvp_f.col(3).setZero();
    double T_bvp = (bvp_f.col(0) - bvp_i.col(0)).norm() / vmax_;
    Piece<7>::CoefficientMat coeffMat;
    double max_omega = 0;
    std::cout << "222"  << std::endl;

    do {
      T_bvp += 1.0;
    std::cout << "000"  << std::endl;

      bvp_f.col(0) = car_p_ + car_v_ * T_bvp;
    std::cout << "111"  << std::endl;

      bvp(T_bvp, bvp_i, bvp_f, coeffMat);
    std::cout << "ppp"  << std::endl;

      std::vector<double> durs{T_bvp};
    std::cout << "2211"  << std::endl;
      std::vector<Piece<7>::CoefficientMat> coeffs{coeffMat};
    std::cout << "22222"  << std::endl;

      Trajectory<7> traj(durs, coeffs);
    std::cout << "22233"  << std::endl;

      max_omega = getMaxOmega(traj);
    } while (max_omega > 1.5 * omega_max_);
    std::cout << "333"  << std::endl;

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
  std::cout << "initial guess >>> t: " << expC2(t) << std::endl;
  std::cout << "initial guess >>> tail_f: " << tail_f << std::endl;

  // NOTE optimization
  lbfgs::lbfgs_parameter_t lbfgs_params;
  lbfgs::lbfgs_load_default_parameters(&lbfgs_params);
  lbfgs_params.mem_size = 32;
  lbfgs_params.past = 3;
  lbfgs_params.g_epsilon = 0.0;
  lbfgs_params.min_step = 1e-16;
  lbfgs_params.delta = 1e-4;
  lbfgs_params.line_search_type = 0;
  double minObjective;

  int opt_ret = 0;

  auto tic = std::chrono::steady_clock::now();
  tictoc_innerloop_ = 0;
  tictoc_integral_ = 0;
  while (cost_lock_.test_and_set())
    ;      
  iter_times_ = 0;
  opt_ret = lbfgs::lbfgs_optimize(dim_t_ + 3 * dim_p_ + 1, x_, &minObjective,
                                  &objectiveFuncLanding, nullptr,
                                  &earlyExitLanding, this, &lbfgs_params);

  auto toc = std::chrono::steady_clock::now();
  cost_lock_.clear();

  std::cout << "\033[32m>ret: " << opt_ret << "\033[0m" << std::endl;
  dashboard_cost_print();

  // std::cout << "innerloop costs: " << tictoc_innerloop_ * 1e-6 << "ms" << std::endl;
  // std::cout << "integral costs: " << tictoc_integral_ * 1e-6 << "ms" << std::endl;
  std::cout << "optmization costs: " << (toc - tic).count() * 1e-6 << "ms" << std::endl;
  // std::cout << "\033[32m>iter times: " << iter_times_ << "\033[0m" << std::endl;
  if (pause_debug_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }
  if (opt_ret < 0) {
    delete[] x_;
    return false;
  }
  double dT = expC2(t);
  double T = N_ * dT;
  Eigen::MatrixXd tailS(3, 4);
  tailS.col(0) = car_p_ + car_v_ * T + tail_q_v_ * robot_l_;
  tailS.col(1) = land_v_;
  tailS.col(2) = forward_thrust(tail_f) * tail_q_v_ + g_;
  tailS.col(3).setZero();
  // std::cout << "tail thrust: " << forward_thrust(tail_f) << std::endl;
  // std::cout << tailS << std::endl;
  minco_s4u_opt_.generate(initS_, tailS, P, dT);
  traj = minco_s4u_opt_.getTraj();

  std::cout << "maxOmega: " << getMaxOmega(traj) << std::endl;
  std::cout << "maxThrust: " << traj.getMaxThrust() << std::endl;

  init_traj_ = traj;
  init_tail_f_ = tail_f;
  initial_guess_ = true;
  delete[] x_;
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
  if (obj.pause_debug_) {
    const double& t = x[0];
    Eigen::Map<const Eigen::MatrixXd> P(x + obj.dim_t_, 3, obj.dim_p_);
    const double& tail_f = x[obj.dim_t_ + obj.dim_p_ * 3];

    double dT = expC2(t);
    double T = obj.N_ * dT;

    Eigen::MatrixXd tailS(3, 4);
    tailS.col(0) = car_p_ + car_v_ * T + tail_q_v_ * obj.robot_l_;
    tailS.col(1) = land_v_;
    tailS.col(2) = forward_thrust(tail_f) * tail_q_v_ + g_;
    tailS.col(3).setZero();

    obj.minco_s4u_opt_.generate(obj.initS_, tailS, P, dT);
    auto traj = obj.minco_s4u_opt_.getTraj();
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
  // std::cout << "damn" << std::endl;
  TrajOpt& obj = *(TrajOpt*)ptrObj;
  obj.iter_times_++;
  obj.clear_cost_rec();

  //! 1. fetch opt varaibles from x_
  const double& t = x[0];
  double& gradt = grad[0];
  Eigen::Map<const Eigen::MatrixXd> P(x + obj.dim_t_, 3, obj.dim_p_);
  Eigen::Map<Eigen::MatrixXd> gradP(grad + obj.dim_t_, 3, obj.dim_p_);
  const double& tail_f = x[obj.dim_t_ + obj.dim_p_ * 3];
  double& grad_f = grad[obj.dim_t_ + obj.dim_p_ * 3];

  //! 2. reform P & T
  double dT = expC2(t);

  //! 3. calculate ternimal state
  Eigen::MatrixXd tailS(3, 4);
  tailS.col(0) = car_p_ + car_v_ * obj.N_ * dT + tail_q_v_ * obj.robot_l_;
  tailS.col(1) = land_v_;
  tailS.col(2) = forward_thrust(tail_f) * tail_q_v_ + g_;
  tailS.col(3).setZero();

  //! 4. generate minco using P & T
  auto tic = std::chrono::steady_clock::now();
  obj.minco_s4u_opt_.generate(obj.initS_, tailS, P, dT);

  double cost = obj.minco_s4u_opt_.getTrajSnapCost();
  obj.minco_s4u_opt_.calGrads_CT();
  obj.cost_snap_rec_ = cost;

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
  toc = std::chrono::steady_clock::now();
  tictoc_innerloop_ += (toc - tic).count();
  // std::cout << "cost of penalty: " << cost - cost_with_only_energy << std::endl;

  obj.minco_s4u_opt_.gdT += obj.minco_s4u_opt_.gdTail.col(0).dot(obj.N_ * car_v_);

  double grad_thrust = obj.minco_s4u_opt_.gdTail.col(2).dot(tail_q_v_);
  addLayerThrust(tail_f, grad_thrust, grad_f);

  //! 7. propogate gradient to opt variable P & tau
  obj.minco_s4u_opt_.gdT += obj.rhoT_;
  cost += obj.rhoT_ * dT;
  gradt = obj.minco_s4u_opt_.gdT * gdT2t(t);
  obj.cost_t_rec_ = obj.rhoT_ * dT;

  gradP = obj.minco_s4u_opt_.gdP;

  return cost;
}
// !SECTION object function


void TrajOpt::addTimeIntPenaltyLanding(double& cost) {
  // cost of one inner sample points
  double cost_inner;
  // state of innner sample point 
  Eigen::Vector3d pos, vel, acc, jer, snp;
  // gradient of the state of innner sample point 
  Eigen::Vector3d grad_p, grad_v, grad_a, grad_j;
  // derivatives of the time base
  Eigen::Matrix<double, 8, 1> beta0, beta1, beta2, beta3, beta4;
  // sample time step; alpha; trapezoidal integral weights
  double step, alpha, omg;
  // current sample time within the segment
  double st = 0.0;
  // gradient propogate to C & T of each segment
  Eigen::Matrix<double, 8, 3> gradViola_c;
  double gradViola_t;
  // exact term cost and gradient
  double cost_vel, cost_thrust, cost_tmp;
  Eigen::Vector3d grad_vel, grad_thrust, grad_tmp, grad_tmp2, grad_tmp3;

  int innerLoop = K_ + 1;
  step = minco_s4u_opt_.t(1) / K_;

  for (int j = 0; j < innerLoop; ++j) {
    beta0 = cal_timebase_snap(0, st);
    beta1 = cal_timebase_snap(1, st);
    beta2 = cal_timebase_snap(2, st);
    beta3 = cal_timebase_snap(3, st);
    beta4 = cal_timebase_snap(4, st);
    alpha = 1.0 / K_ * j;
    omg = (j == 0 || j == innerLoop - 1) ? 0.5 : 1.0;

    for (int i = 0; i < N_; ++i) {
      const auto& c = minco_s4u_opt_.c.block<8, 3>(i * 8, 0);

      pos = c.transpose() * beta0;
      vel = c.transpose() * beta1;
      acc = c.transpose() * beta2;
      jer = c.transpose() * beta3;
      snp = c.transpose() * beta4;

      grad_p.setZero();
      grad_v.setZero();
      grad_a.setZero();
      grad_j.setZero();
      cost_inner = 0.0;

      // if (grad_cost_floor(pos, grad_tmp, cost_tmp)) {
      //   grad_p += grad_tmp;
      //   cost_inner += cost_tmp;
      // }

      if (grad_cost_v(vel, grad_vel, cost_vel)) {
        grad_v += grad_vel;
        cost_inner += cost_vel;
        cost_v_rec_ += omg * step * cost_vel;
      }

      if (grad_cost_thrust(acc, grad_thrust, cost_thrust)) {
        grad_a += grad_thrust;
        cost_inner += cost_thrust;
        cost_thrust_rec_ += omg * step * cost_thrust;
      }

      if (grad_cost_omega(acc, jer, grad_tmp, grad_tmp2, cost_tmp)) {
        grad_a += grad_tmp;
        grad_j += grad_tmp2;
        cost_inner += cost_tmp;
        cost_omega_rec_ += omg * step * cost_tmp;
      }

      // if (grad_cost_omega_yaw(acc, jer, grad_tmp, grad_tmp2, cost_tmp)) {
      //   grad_a += grad_tmp;
      //   grad_j += grad_tmp2;
      //   cost_inner += cost_tmp;
      // }

      double dur2now = (i + alpha) * minco_s4u_opt_.t(1);
      Eigen::Vector3d car_p = car_p_ + car_v_ * dur2now;
      if (grad_cost_perching_collision(pos, acc, car_p,
                                       grad_tmp, grad_tmp2, grad_tmp3,
                                       cost_tmp)) {
        grad_p += grad_tmp;
        grad_a += grad_tmp2;
        cost_inner += cost_tmp;
        cost_perching_collision_rec_ += omg * step * cost_tmp;
      }
      double grad_car_t = grad_tmp3.dot(car_v_);

      gradViola_c = beta0 * grad_p.transpose();
      gradViola_t = grad_p.transpose() * vel;
      gradViola_c += beta1 * grad_v.transpose();
      gradViola_t += grad_v.transpose() * acc;
      gradViola_c += beta2 * grad_a.transpose();
      gradViola_t += grad_a.transpose() * jer;
      gradViola_c += beta3 * grad_j.transpose();
      gradViola_t += grad_j.transpose() * snp;
      gradViola_t += grad_car_t;

      minco_s4u_opt_.gdC.block<8, 3>(i * 8, 0) += omg * step * gradViola_c;
      minco_s4u_opt_.gdT += omg * (cost_inner / K_ + alpha * step * gradViola_t);
      minco_s4u_opt_.gdT += i * omg * step * grad_car_t;
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
  static double z_floor = 0.4;
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