#pragma once
#include <Eigen/Geometry>
#include <mutex>
#include <atomic>
#include "util_gym/util_gym.hpp"
#include "parameter_server/parameter_server.hpp"
#include "rotation_util/rotation_util.hpp"
#include "target_ekf/target_ekf_car.hpp"

namespace ekf_server{
using rot_util = rotation_util::RotUtil;

class EKFServer{
 private:
  std::shared_ptr<EkfCar> ekf_ptr_;
  std::shared_ptr<parameter_server::ParaeterSerer> para_ptr_;
  // std::mutex mutex_;
  std::atomic_flag ekf_lock_ = ATOMIC_FLAG_INIT;
  bool is_land_mode_ = false;
  double filtered_v_k_land_, filtered_v_k_track_;

  // double predict_dt_;
  TimePoint last_update_stamp_;
  bool has_last_update_ = false;
  Eigen::Vector3d filtered_v_;
  
  // use diff of theta input data to get omega
  double omega_, omega_max_;
  double omega_filter_t_;

  inline void update_omega(const double old_theta, const double cur_theta, const double dt){
    if (has_last_update_){
      omega_ = (1 - omega_filter_t_) * omega_ + 
                omega_filter_t_ * (cur_theta - old_theta) / dt;
      if (omega_ > omega_max_){
        omega_ = omega_max_;
      }
      if (omega_ < -omega_max_){
        omega_ = -omega_max_;
      }
    }else{
      omega_ = 0.0;
    }
  }
  inline double get_omega(){
    return omega_;
  }

  // do the prediction before update
  void predict_state(Eigen::Vector3d& p_res, 
                     double& v_res, 
                     double& theta_res, 
                     const TimePoint& predict_tiemstamp = TimeNow()){
    if (!has_last_update_){
      INFO_MSG_RED("[ekf_server] error! predict without update");
      return;
    }
    double dt = durationSecond(predict_tiemstamp, last_update_stamp_);
    ekf_ptr_->predict(dt);

    // int loop_time = floor(dt / predict_dt_);
    // for (int i = 0; i < loop_time; i++){
    //   ekf_ptr_->predict(predict_dt_);
    // }
    // double remain_dt = dt - loop_time * predict_dt_;
    // if (remain_dt > 1e-5){
    //   ekf_ptr_->predict(remain_dt);
    // }
    p_res = ekf_ptr_->pos();
    v_res = ekf_ptr_->vel();
    theta_res = ekf_ptr_->theta();
  }

 public:
  bool update_data_received(){
    return has_last_update_;
  }
  inline void set_land_mode(){
    is_land_mode_ = true;
  }

  inline void set_track_mode(){
    is_land_mode_ = false;
  }
  // measurement: [px, py, pz]
  void update_p_state(const Eigen::Vector3d& p_wt, const TimePoint& update_tiemstamp, const double R_coef = 1.0){
    while (ekf_lock_.test_and_set())
      ;

    Eigen::Vector3d p_pre;
    double v_pre, theta_pre;
    
    double old_theta = ekf_ptr_->theta();
    if (has_last_update_){
      predict_state(p_pre, v_pre, theta_pre, update_tiemstamp);
    }
    // INFO_MSG("update with p: " << p_wt.transpose());
    ekf_ptr_->update_p(p_wt, R_coef);
    // INFO_MSG("after update p: " << ekf_ptr_->pos().transpose());
    double cur_theta = ekf_ptr_->theta();
    update_omega(old_theta, cur_theta, durationSecond(update_tiemstamp, last_update_stamp_));
    // INFO_MSG("update_tiemstamp: " <<update_tiemstamp.time_since_epoch().count()<<", last_update_stamp_: "<<last_update_stamp_.time_since_epoch().count());
    last_update_stamp_ = update_tiemstamp;
    has_last_update_ = true;
    ekf_lock_.clear();
  }

  // measurement: [px, py, pz], vel obtained by difference
  void update_p_state_diff_v(const Eigen::Vector3d& p_wt, const TimePoint& update_tiemstamp, const double R_coef = 1.0){
    while (ekf_lock_.test_and_set())
      ;

    Eigen::Vector3d p_pre;
    double v_pre, theta_pre;

    static bool has_last_p_wt = false;
    static Eigen::Vector3d last_p_wt = Eigen::Vector3d::Zero();
    static TimePoint last_p_wt_timestamp;

    Eigen::Vector3d p_wt_v = Eigen::Vector3d::Zero();
    if (has_last_p_wt){
      double update_dt = durationSecond(update_tiemstamp, last_p_wt_timestamp);
      p_wt_v = (p_wt - last_p_wt) / update_dt;
      if (update_dt > 0.5) has_last_p_wt = false;
    }
    last_p_wt = p_wt;
    last_p_wt_timestamp = update_tiemstamp;
    if (!has_last_p_wt)
    {
      has_last_p_wt = true;
      ekf_lock_.clear();
      return;
    }
    if (is_land_mode_){
      filtered_v_ = (1 - filtered_v_k_land_) * filtered_v_ + filtered_v_k_land_ * p_wt_v;
    }else{
      filtered_v_ = (1 - filtered_v_k_track_) * filtered_v_ + filtered_v_k_track_ * p_wt_v;
    }

    double old_theta = ekf_ptr_->theta();
    if (has_last_update_){
      predict_state(p_pre, v_pre, theta_pre, update_tiemstamp);
    }

    // INFO_MSG("update with p: " << p_wt.transpose());
    Eigen::Vector4d z(p_wt.x(), p_wt.y(), p_wt.z(), filtered_v_.head(2).norm());
    ekf_ptr_->update_pv(z, R_coef);
    // INFO_MSG("after update p: " << ekf_ptr_->pos().transpose());
    double cur_theta = ekf_ptr_->theta();
    update_omega(old_theta, cur_theta, durationSecond(update_tiemstamp, last_update_stamp_));
    // INFO_MSG("update_tiemstamp: " <<update_tiemstamp.time_since_epoch().count()<<", last_update_stamp_: "<<last_update_stamp_.time_since_epoch().count());
    last_update_stamp_ = update_tiemstamp;
    last_p_wt = p_wt;
    has_last_update_ = true;
    ekf_lock_.clear();
  }

  // measurement: [px, py, pz, theta], vel obtained by difference
  void update_ptheta_diff_v(const Eigen::Vector3d& p_wt, const double theta, const TimePoint& update_tiemstamp, const double R_coef = 1.0){
    while (ekf_lock_.test_and_set())
      ;

    Eigen::Vector3d p_pre;
    double v_pre, theta_pre;

    static bool has_last_p_wt = false;
    static Eigen::Vector3d last_p_wt = Eigen::Vector3d::Zero();
    static TimePoint last_p_wt_timestamp;

    Eigen::Vector3d p_wt_v = Eigen::Vector3d::Zero();
    if (has_last_p_wt){
      double update_dt = durationSecond(update_tiemstamp, last_p_wt_timestamp);
      p_wt_v = (p_wt - last_p_wt) / update_dt;
      if (update_dt > 0.5) has_last_p_wt = false;
    }
    last_p_wt = p_wt;
    last_p_wt_timestamp = update_tiemstamp;
    if (!has_last_p_wt)
    {
      has_last_p_wt = true;
      ekf_lock_.clear();
      return;
    }
    if (is_land_mode_){
      filtered_v_ = (1 - filtered_v_k_land_) * filtered_v_ + filtered_v_k_land_ * p_wt_v;
    }else{
      filtered_v_ = (1 - filtered_v_k_track_) * filtered_v_ + filtered_v_k_track_ * p_wt_v;
    }
    
    double old_theta = ekf_ptr_->theta();
    if (has_last_update_){
      predict_state(p_pre, v_pre, theta_pre, update_tiemstamp);
    }

    // INFO_MSG("update with p: " << p_wt.transpose());
    Eigen::Matrix<double, 5, 1> z;
    z << p_wt.x(), p_wt.y(), p_wt.z(), theta, filtered_v_.head(2).norm();
    ekf_ptr_->update_all(z, R_coef);
    // INFO_MSG("after update p: " << ekf_ptr_->pos().transpose());
    double cur_theta = ekf_ptr_->theta();
    update_omega(old_theta, cur_theta, durationSecond(update_tiemstamp, last_update_stamp_));
    // INFO_MSG("update_tiemstamp: " <<update_tiemstamp.time_since_epoch().count()<<", last_update_stamp_: "<<last_update_stamp_.time_since_epoch().count());
    last_update_stamp_ = update_tiemstamp;
    last_p_wt = p_wt;
    has_last_update_ = true;
    ekf_lock_.clear();
  }

  // measurement: [px, py, pz, theta]
  void update_ptheta_state(const Eigen::Vector3d& p_wt, const double theta, const TimePoint& update_tiemstamp, const double R_coef = 1.0){
    while (ekf_lock_.test_and_set())
      ;
    Eigen::Vector3d p_pre;
    double v_pre, theta_pre;
    double old_theta = ekf_ptr_->theta();
    if (has_last_update_){
      predict_state(p_pre, v_pre, theta_pre, update_tiemstamp);
    }
    Eigen::Matrix<double, 4, 1> z;
    z << p_wt.x(), p_wt.y(), p_wt.z(), theta;
    ekf_ptr_->update_ptheta(z, R_coef);
    double cur_theta = ekf_ptr_->theta();
    update_omega(old_theta, cur_theta, durationSecond(update_tiemstamp, last_update_stamp_));
    last_update_stamp_ = update_tiemstamp;
    has_last_update_ = true;
    ekf_lock_.clear();
  }

  // get prediction upon the clone ekf obj
  void get_predict_state(Eigen::Vector3d& p_res, 
                         double& v_res, 
                         double& theta_res, 
                         double& omega_res,
                         const TimePoint& predict_tiemstamp = TimeNow()){
    if (!has_last_update_){
      INFO_MSG_RED("[ekf_server] error! predict without update");
      return;
    }
    // INFO_MSG("get mutex begin");

    EkfCar tmp_ekf;
    double dt;
    {
      while (ekf_lock_.test_and_set())
        ;

      tmp_ekf.Q_ = ekf_ptr_->Q_;
      tmp_ekf.R_ = ekf_ptr_->R_;
      tmp_ekf.x_ = ekf_ptr_->x_;
      tmp_ekf.Sigma_ = ekf_ptr_->Sigma_;

      // tmp_ekf = *ekf_ptr_;
      // INFO_MSG("ekf_ptr_: "<<ekf_ptr_->pos().transpose());
      // INFO_MSG("tmp_ekf: "<<tmp_ekf.pos());
      dt = durationSecond(predict_tiemstamp, last_update_stamp_);
      ekf_lock_.clear();
    }

    // INFO_MSG("get mutex done");
    if (dt <= 0.0){
      INFO_MSG_RED("[ekf_server] error! predict timestamp too old!");
      p_res = tmp_ekf.pos();
      v_res = tmp_ekf.vel();
      theta_res = tmp_ekf.theta();
      omega_res = get_omega();
      return;
    }

    tmp_ekf.predict(dt);

    p_res = tmp_ekf.pos();
    v_res = tmp_ekf.vel();
    theta_res = tmp_ekf.theta();
    omega_res = get_omega();
  }

  EKFServer(std::shared_ptr<parameter_server::ParaeterSerer> para_ptr): para_ptr_(para_ptr){
    ekf_ptr_ = std::make_shared<EkfCar>();
    para_ptr_->get_para("prediction/omega_max", omega_max_);
    para_ptr_->get_para("prediction/omega_filter_t", omega_filter_t_);
    para_ptr_->get_para("EKF/Qx", ekf_ptr_->Q_(0));
    para_ptr_->get_para("EKF/Qy", ekf_ptr_->Q_(1));
    para_ptr_->get_para("EKF/Qz", ekf_ptr_->Q_(2));
    para_ptr_->get_para("EKF/Qv", ekf_ptr_->Q_(3));
    para_ptr_->get_para("EKF/Qtheta", ekf_ptr_->Q_(4));
    para_ptr_->get_para("EKF/Rx", ekf_ptr_->R_(0));
    para_ptr_->get_para("EKF/Ry", ekf_ptr_->R_(1));
    para_ptr_->get_para("EKF/Rz", ekf_ptr_->R_(2));
    para_ptr_->get_para("EKF/Rv", ekf_ptr_->R_(3));
    para_ptr_->get_para("EKF/Rtheta", ekf_ptr_->R_(4));
    para_ptr_->get_para("EKF/filtered_v_k_track", filtered_v_k_track_);
    para_ptr_->get_para("EKF/filtered_v_k_land", filtered_v_k_land_);

    filtered_v_.setZero();
  }

};

} // namespace ekf_server
