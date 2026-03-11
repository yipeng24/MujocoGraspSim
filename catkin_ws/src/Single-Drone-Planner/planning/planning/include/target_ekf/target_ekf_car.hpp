#pragma once
#include <Eigen/Geometry>
#include "util_gym/util_gym.hpp"
#include "rotation_util/rotation_util.hpp"

namespace ekf_server{
using rot_util = rotation_util::RotUtil;
// This EKF class uses bicycle model
class EkfCar {
 public:
   // states: x, y, z, v, yaw
  Eigen::VectorXd x_;
  Eigen::MatrixXd Sigma_;

 public:
  Eigen::VectorXd Q_, R_;

  EkfCar(){
    x_.setZero(5);
    Sigma_.setZero(5, 5);
    Q_.setOnes(5);
    R_.setOnes(5);
  }

  EkfCar(const EkfCar& obj){
    x_ = obj.x_;
    Sigma_ = obj.Sigma_;
    Q_ = obj.Q_;
    R_ = obj.R_;
  }

  EkfCar& operator = (const EkfCar& obj){
    x_ = obj.x_;
    Sigma_ = obj.Sigma_;
    Q_ = obj.Q_;
    R_ = obj.R_;
    return *this;
  }

  inline void predict(const double& dt) {
    Eigen::MatrixXd G  = get_G(x_, dt);
    Eigen::MatrixXd Qt = get_Qt(dt);
    x_ = transit(x_, dt);
    Sigma_ = G * Sigma_ * G.transpose() + Qt;
    return;
  }

  // State transit
  inline Eigen::VectorXd transit(const Eigen::VectorXd& x, const double& dt){
    Eigen::VectorXd x_res;
    x_res.resize(5);
    double px_in    = x(0);
    double py_in    = x(1);
    double pz_in    = x(2);
    double v_in     = x(3);
    double theta_in = x(4);
    x_res(0) = px_in + v_in*cos(theta_in)*dt;
    x_res(1) = py_in + v_in*sin(theta_in)*dt;
    x_res(2) = pz_in;
    x_res(3) = v_in;
    x_res(4) = theta_in;
    return x_res;
  }

  // get the Jacobi of the transit matrix
  inline Eigen::MatrixXd get_G(const Eigen::VectorXd& x, const double& dt){
    Eigen::Matrix<double, 5, 5> G;
    double v_in     = x(3);
    double theta_in = x(4);
    Eigen::Matrix<double, 1, 5> dpxdx, dpydx, dpzdx, dvdx, dthetadx;
    dpxdx << 1, 0, 0, cos(theta_in)*dt, -sin(theta_in)*v_in*dt;
    dpydx << 0, 1, 0, sin(theta_in)*dt,  cos(theta_in)*v_in*dt;
    dpzdx << 0, 0, 1, 0, 0;
    dvdx  << 0, 0, 0, 1, 0;
    dthetadx << 0, 0, 0, 0, 1;
    G.row(0) = dpxdx;
    G.row(1) = dpydx;
    G.row(2) = dpzdx;
    G.row(3) = dvdx;
    G.row(4) = dthetadx;
    return G;
  }

  // get the process noise matrix
  inline Eigen::MatrixXd get_Qt(const double& dt){
    double dt2 = dt * dt / 2.0;
    Eigen::Matrix<double, 5, 5> Qt;
    Qt.setZero();
    Qt(0, 0) = Q_(0) * dt2;
    Qt(1, 1) = Q_(1) * dt2;
    Qt(2, 2) = Q_(2) * dt2;
    Qt(3, 3) = Q_(3) * dt;
    Qt(4, 4) = Q_(4) * dt;
    Qt = Qt;
    return Qt;
  }

  // update with [px, py, pz] measurement
  inline void update_p(const Eigen::Matrix<double, 3, 1>& z, const double R_coef = 1.0) {
    Eigen::MatrixXd Rt, C;
    Rt.setZero(3, 3);
    C.setZero(3, 5);
    Rt(0, 0) = R_(0);
    Rt(1, 1) = R_(1);
    Rt(2, 2) = R_(2);
    Rt = Rt * R_coef;
    C(0, 0)  = 1.0;
    C(1, 1)  = 1.0;
    C(2, 2)  = 1.0;
    Eigen::MatrixXd K;
    K = Sigma_ * C.transpose() * (C * Sigma_ * C.transpose() + Rt).inverse();
    x_ = x_ + K * (z - C * x_);
    Sigma_ = Sigma_ - K * C * Sigma_;
    rot_util::rad_limit(x_(4));
  }

  // update with [px, py, pz, v] measurement
  inline void update_pv(const Eigen::Matrix<double, 4, 1>& z, const double R_coef = 1.0) {
    Eigen::MatrixXd Rt, C;
    Rt.setZero(4, 4);
    C.setZero(4, 5);
    Rt(0, 0) = R_(0);
    Rt(1, 1) = R_(1);
    Rt(2, 2) = R_(2);
    Rt(3, 3) = R_(3);
    Rt = Rt * R_coef;
    C(0, 0)  = 1.0;
    C(1, 1)  = 1.0;
    C(2, 2)  = 1.0;
    C(3, 3)  = 1.0;
    Eigen::MatrixXd K;
    K = Sigma_ * C.transpose() * (C * Sigma_ * C.transpose() + Rt).inverse();
    x_ = x_ + K * (z - C * x_);
    Sigma_ = Sigma_ - K * C * Sigma_;
    rot_util::rad_limit(x_(4));
  }

  // update with [px, py, pz, theta] measurement
  inline void update_ptheta(const Eigen::Matrix<double, 4, 1>& z, const double R_coef = 1.0) {
    Eigen::MatrixXd Rt, C;
    Rt.setZero(4, 4);
    C.setZero(4, 5);
    Rt(0, 0) = R_(0);
    Rt(1, 1) = R_(1);
    Rt(2, 2) = R_(2);
    Rt(3, 3) = R_(4);
    Rt = Rt * R_coef;
    C(0, 0)  = 1.0;
    C(1, 1)  = 1.0;
    C(2, 2)  = 1.0;
    C(3, 4)  = 1.0;
    Eigen::MatrixXd K;
    K = Sigma_ * C.transpose() * (C * Sigma_ * C.transpose() + Rt).inverse();
    x_ = x_ + K * (z - C * x_);
    Sigma_ = Sigma_ - K * C * Sigma_;
    rot_util::rad_limit(x_(4));
  }

  // update with [px, py, pz, v, theta] measurement
  inline void update_all(const Eigen::Matrix<double, 5, 1>& z, const double R_coef = 1.0) {
    Eigen::MatrixXd Rt, C;
    Rt.setZero(5, 5);
    C.setZero(5, 5);
    Rt(0, 0) = R_(0);
    Rt(1, 1) = R_(1);
    Rt(2, 2) = R_(2);
    Rt(3, 3) = R_(3);
    Rt(4, 4) = R_(4);
    Rt = Rt * R_coef;
    C.setIdentity(5, 5);
    Eigen::MatrixXd K;
    K = Sigma_ * C.transpose() * (C * Sigma_ * C.transpose() + Rt).inverse();
    x_ = x_ + K * (z - C * x_);
    Sigma_ = Sigma_ - K * C * Sigma_;
    rot_util::rad_limit(x_(4));
  }

  inline void reset(const Eigen::Vector3d& z) {
    if (z.size() != 5) INFO_MSG_RED("[ekf_car] reset measuremet size error!");
    x_ = z;
    Sigma_.setZero();
  }

  inline void set_p(const Eigen::Vector3d& p){
    x_.head(3) = p;
  }
  inline void set_v(const double& v){
    x_(3) = v;
  }
  inline void set_theta(const double& theta){
    x_(4) = theta;
  }
  inline void reset_vel(){
    x_(3) = 0.0;
  }

  inline const Eigen::Vector3d pos() const {
    return x_.head(3);
  }
  inline const double vel() const {
    return x_(3);
  }
  inline const double theta() const {
    return x_(4);
  }

};



} // namespace ekf_server
