#pragma once
#include <Eigen/Geometry>
#include "util_gym/util_gym.hpp"
#include "parameter_server/parameter_server.hpp"
#include "rotation_util/rotation_util.hpp"

namespace ekf_server{
using rot_util = rotation_util::RotUtil;
// here we just use no-model ekf for data fusion, the filtering function is more dependent on user-defined low-pass filter
struct Ekf {
  Eigen::MatrixXd A, B, C;
  Eigen::MatrixXd Qt, Rt;
  Eigen::MatrixXd Sigma, K;
  Eigen::VectorXd x;

  // states: x, y, z, v, yaw

  Ekf(){
    A.setIdentity(9, 9);
    C.setIdentity(9, 9);
    Sigma.setZero(9, 9);
    K = C;
    Qt.setIdentity(9, 9);
    Rt.setIdentity(9, 9);
    Qt(0, 0) = 1;    // x
    Qt(1, 1) = 1;    // y
    Qt(2, 2) = 1;    // z
    Qt(3, 3) = 3;    // vx
    Qt(4, 4) = 3;    // vy
    Qt(5, 5) = 3;  // vz
    Qt(6, 6) = 0.1;    // roll
    Qt(7, 7) = 0.1;    // pitch
    Qt(8, 8) = 0.1;  // yaw

    Rt(0, 0) = 0.1; // x
    Rt(1, 1) = 0.1; // y
    Rt(2, 2) = 0.1; //z
    Rt(3, 3) = 2;   //vx
    Rt(4, 4) = 2;   //vy
    Rt(5, 5) = 2;   //vz
    Rt(6, 6) = 0.01;//roll
    Rt(7, 7) = 0.01;//pitch
    Rt(8, 8) = 0.01;//yaw
    x.setZero(9);
  }
  inline void predict(const double& dt) {
    A(0, 3) = dt;
    A(1, 4) = dt;
    A(2, 5) = dt;
    x = A * x;
    Sigma = A * Sigma * A.transpose() + Qt*dt;
    return;
  }
  inline void reset(const Eigen::Vector3d& z, const Eigen::Vector3d& z_v, const Eigen::Vector3d& z_rpy) {
    x.setZero();
    x.head(3) = z;
    x.middleRows(3, 3) = z_v;
    x.tail(3) = z_rpy;
    Sigma.setZero();
  }
  inline void reset_vel(){
    x.middleRows(3, 3) = Eigen::Vector3d::Zero();
  }
  inline bool update(const Eigen::Vector3d& z, const Eigen::Vector3d& z_v, const Eigen::Vector3d& z_rqp, 
                     const double R_coef = 1.0) {
    K = Sigma * C.transpose() * (C * Sigma * C.transpose() + Rt * R_coef).inverse();
    Eigen::VectorXd zz(9);
    zz.head(3) = z;
    zz.middleRows(3, 3) = z_v;
    zz.tail(3) = z_rqp;
    Eigen::VectorXd x_tmp = x + K * (zz - C * x);
    // NOTE check valid
    // TODO
    Eigen::Vector3d d_rpy = x.tail(3) - z_rqp;
    x.tail(3).x() = d_rpy.x() > M_PI ? x.tail(3).x() - 2 * M_PI : x.tail(3).x();
    x.tail(3).y() = d_rpy.y() > M_PI ? x.tail(3).y() - 2 * M_PI : x.tail(3).y();
    x.tail(3).z() = d_rpy.z() > M_PI ? x.tail(3).z() - 2 * M_PI : x.tail(3).z();
    x.tail(3).x() = d_rpy.x() < -M_PI ? x.tail(3).x() + 2 * M_PI : x.tail(3).x();
    x.tail(3).y() = d_rpy.y() < -M_PI ? x.tail(3).y() + 2 * M_PI : x.tail(3).y();
    x.tail(3).z() = d_rpy.z() < -M_PI ? x.tail(3).z() + 2 * M_PI : x.tail(3).z();
    x = x + K * (zz - C * x);
    Sigma = Sigma - K * C * Sigma;
    return true;
  }
  inline const Eigen::Vector3d pos() const {
    return x.head(3);
  }
  inline const Eigen::Vector3d vel() const {
    return x.middleRows(3, 3);
  }
  inline const Eigen::Vector3d rpy() const {
    return x.tail(3);
  }
  inline const double yaw() const {
    return x(8);
  }
};



} // namespace ekf_server
