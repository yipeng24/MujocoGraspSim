#pragma once
#include <Eigen/Geometry>
#include <mutex>
#include <atomic>
#include "util_gym/util_gym.hpp"
#include "parameter_server/parameter_server.hpp"
#include "rotation_util/rotation_util.hpp"
#include "target_ekf/target_ekf_99.hpp"

namespace ekf_server{
using rot_util = rotation_util::RotUtil;

class EKFServer{
 private:
  std::shared_ptr<Ekf> ekf_ptr_;
  std::shared_ptr<parameter_server::ParaeterSerer> para_ptr_;
  // std::mutex mutex_;
  std::atomic_flag ekf_lock_ = ATOMIC_FLAG_INIT;
  bool is_land_mode_ = false;
  double filtered_v_k_land_, filtered_v_k_track_, filtered_q_k_, filtered_q_diff_k_;

  // double predict_dt_;
  TimePoint last_update_stamp_;
  bool has_last_update_ = false;
  Eigen::Vector3d filtered_v_;
  Eigen::Quaterniond filtered_q_;
  
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

  // use diff of theta input data to get acc
  Eigen::Vector3d acc_;
  double acc_max_;
  double acc_filter_t_;

  inline void update_acc(const Eigen::Vector3d& old_v, const Eigen::Vector3d& cur_v, const double dt){
    if (has_last_update_){
      acc_ = (1 - acc_filter_t_) * acc_ + 
                acc_filter_t_ * (cur_v - old_v) / dt;
      if (acc_.x() > acc_max_){
        acc_.x() = acc_max_;
      }
      if (acc_.x() < -acc_max_){
        acc_.x() = -acc_max_;
      }
      if (acc_.y() > acc_max_){
        acc_.y() = acc_max_;
      }
      if (acc_.y() < -acc_max_){
        acc_.y() = -acc_max_;
      }
      acc_.z() = 0.0;
    }else{
      acc_.setZero();
    }
  }
  inline Eigen::Vector3d get_acc(){
    return acc_;
  }

  // do the prediction before update
  void predict_state(Eigen::Vector3d& p_res, 
                     Eigen::Vector3d& v_res, 
                     Eigen::Vector3d& rpy_res, 
                     const TimePoint& predict_tiemstamp = TimeNow()){
    if (!has_last_update_){
      INFO_MSG_RED("[ekf_server] error! predict without update");
      return;
    }
    double dt = durationSecond(predict_tiemstamp, last_update_stamp_);
    ekf_ptr_->predict(dt);

    p_res = ekf_ptr_->pos();
    v_res = ekf_ptr_->vel();
    rpy_res = ekf_ptr_->rpy();
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

  // measurement: [px, py, pz], vel obtained by difference
  bool update_p_state_diff_v(const Eigen::Vector3d& p_wt, const Eigen::Vector3d& dp, const TimePoint& update_tiemstamp, const bool& type_switch = false, const double R_coef = 1.0){
    while (ekf_lock_.test_and_set())
      ;
    bool status_ok = true;
    static bool has_last_p_wt = false;
    static Eigen::Vector3d last_p_wt = Eigen::Vector3d::Zero();
    static TimePoint last_p_wt_timestamp;

    if (type_switch){
      INFO_MSG("type switch!");
      has_last_p_wt = false;
    }

    Eigen::Vector3d p_wt_v = Eigen::Vector3d::Zero();
    if (has_last_p_wt){
      double update_dt = durationSecond(update_tiemstamp, last_p_wt_timestamp);
      p_wt_v = (p_wt - last_p_wt) / update_dt;
      p_wt_v.z() = 0.0;
      if (update_dt > 0.5) has_last_p_wt = false;
    }
    last_p_wt = p_wt;
    last_p_wt_timestamp = update_tiemstamp;
    if (!has_last_p_wt)
    {
      has_last_p_wt = true;
      ekf_lock_.clear();
      return status_ok;
    }
    // only if having twice the same type observation can do vel estimation update
    if (is_land_mode_){
      filtered_v_ = (1 - filtered_v_k_land_) * filtered_v_ + filtered_v_k_land_ * p_wt_v;
    }else{
      filtered_v_ = (1 - filtered_v_k_track_) * filtered_v_ + filtered_v_k_track_ * p_wt_v;
    }
    Eigen::Quaterniond q = rot_util::yaw2quaternion(atan2(filtered_v_.y(), filtered_v_.x()));
    if (filtered_v_.head(2).norm() > 0.4){
      filtered_q_ = filtered_q_.slerp(filtered_q_diff_k_, q);
    }
    
    Eigen::Vector3d p_pre;
    Eigen::Vector3d v_pre, rpy_pre;
    double old_theta = ekf_ptr_->yaw();
    Eigen::Vector3d old_v = ekf_ptr_->vel();
    if (has_last_update_){
      predict_state(p_pre, v_pre, rpy_pre, update_tiemstamp);
    }

    // INFO_MSG("update with p: " << p_wt.transpose());
    Eigen::Vector3d rpy = rot_util::quaternion2euler(filtered_q_);
    Eigen::Vector3d p_wt_fix = p_wt + filtered_q_ * dp;
    ekf_ptr_->update(p_wt_fix, filtered_v_, rpy, R_coef);
    // INFO_MSG("after update p: " << ekf_ptr_->pos().transpose());
    double cur_theta = ekf_ptr_->yaw();
    double dt = durationSecond(update_tiemstamp, last_update_stamp_);
    update_omega(old_theta, cur_theta, dt);
    Eigen::Vector3d cur_v = ekf_ptr_->vel();
    update_acc(old_v, cur_v, dt);
    Eigen::Vector3d raw_acc = (cur_v - old_v) / dt;
    if (abs(raw_acc.x()) > 15 || abs(raw_acc.y()) > 15){
      status_ok = false;
      INFO_MSG_RED("ACC BOOM!!! " << raw_acc.transpose());
    }
    // INFO_MSG("update_tiemstamp: " <<update_tiemstamp.time_since_epoch().count()<<", last_update_stamp_: "<<last_update_stamp_.time_since_epoch().count());
    last_update_stamp_ = update_tiemstamp;
    has_last_update_ = true;
    ekf_lock_.clear();
    return status_ok;
  }

  // measurement: [px, py, pz, theta], vel obtained by difference
  bool update_ptheta_diff_v(const Eigen::Vector3d& p_wt, const Eigen::Quaterniond& q, const TimePoint& update_tiemstamp, const double R_coef = 1.0){
    while (ekf_lock_.test_and_set())
      ;
    bool status_ok = true;
    static bool has_last_p_wt = false;
    static bool first_update = false;
    static Eigen::Vector3d last_p_wt = Eigen::Vector3d::Zero();
    static TimePoint last_p_wt_timestamp;

    Eigen::Vector3d p_wt_v = Eigen::Vector3d::Zero();
    if (has_last_p_wt){
      double update_dt = durationSecond(update_tiemstamp, last_p_wt_timestamp);
      p_wt_v = (p_wt - last_p_wt) / update_dt;
      // p_wt_v.z() = 0.0;
      if (update_dt > 0.5) has_last_p_wt = false;
    }
    last_p_wt = p_wt;
    last_p_wt_timestamp = update_tiemstamp;
    if (!has_last_p_wt)
    {
      has_last_p_wt = true;
      first_update = true;
      ekf_lock_.clear();
      return status_ok;
    }
    // only if having twice the same type observation can do vel estimation update
    if (is_land_mode_){
      filtered_v_ = (1 - filtered_v_k_land_) * filtered_v_ + filtered_v_k_land_ * p_wt_v;
    }else{
      filtered_v_ = (1 - filtered_v_k_track_) * filtered_v_ + filtered_v_k_track_ * p_wt_v;
    }
    filtered_q_ = filtered_q_.slerp(filtered_q_k_, q);
    
    Eigen::Vector3d p_pre;
    Eigen::Vector3d v_pre, rpy_pre;
    double old_theta = ekf_ptr_->yaw();
    Eigen::Vector3d old_v = ekf_ptr_->vel();
    // INFO_MSG("update with p: " << p_wt.transpose());
    Eigen::Vector3d rpy = rot_util::quaternion2euler(filtered_q_);
    if (first_update){
      INFO_MSG("[ekf] first_update, reset");
      ekf_ptr_->reset(p_wt, filtered_v_, rpy);
    }else{
      if (has_last_update_){
        predict_state(p_pre, v_pre, rpy_pre, update_tiemstamp);
      }
      ekf_ptr_->update(p_wt, filtered_v_, rpy, R_coef);
    }
    
    // INFO_MSG("after update p: " << ekf_ptr_->pos().transpose());
    if (!first_update){
      double cur_theta = ekf_ptr_->yaw();
      double dt = durationSecond(update_tiemstamp, last_update_stamp_);
      update_omega(old_theta, cur_theta, dt);
      Eigen::Vector3d cur_v = ekf_ptr_->vel();
      update_acc(old_v, cur_v, dt);
      Eigen::Vector3d raw_acc = (cur_v - old_v) / dt;
      if (abs(raw_acc.x()) > 15 || abs(raw_acc.y()) > 15){
        status_ok = false;
        INFO_MSG_RED("ACC BOOM!!! " << raw_acc.transpose());
      }
    }

    if (first_update){
      first_update = false;
    }

    // INFO_MSG("update_tiemstamp: " <<update_tiemstamp.time_since_epoch().count()<<", last_update_stamp_: "<<last_update_stamp_.time_since_epoch().count());
    last_update_stamp_ = update_tiemstamp;
    has_last_update_ = true;
    ekf_lock_.clear();
    return status_ok;
  }

  // get prediction upon the clone ekf obj
  void get_predict_state(Eigen::Vector3d& p_res, 
                         Eigen::Vector3d& v_res, 
                         Eigen::Vector3d& a_res, 
                         Eigen::Vector3d& rpy_res, 
                         double& omega_res,
                         const TimePoint& predict_tiemstamp = TimeNow()){
    if (!has_last_update_){
      INFO_MSG_RED("[ekf_server] error! predict without update");
      return;
    }
    // INFO_MSG("get mutex begin");

    Ekf tmp_ekf;
    double dt;
    {
      while (ekf_lock_.test_and_set())
        ;

      tmp_ekf.Qt = ekf_ptr_->Qt;
      tmp_ekf.Rt = ekf_ptr_->Rt;
      tmp_ekf.x = ekf_ptr_->x;
      tmp_ekf.Sigma = ekf_ptr_->Sigma;

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
      rpy_res = tmp_ekf.rpy();
      omega_res = get_omega();
      a_res = get_acc();
      return;
    }

    tmp_ekf.predict(dt);

    p_res = tmp_ekf.pos();
    v_res = tmp_ekf.vel();
    rpy_res = tmp_ekf.rpy();
    omega_res = get_omega();
    a_res = get_acc();
  }


  EKFServer(std::shared_ptr<parameter_server::ParaeterSerer> para_ptr): para_ptr_(para_ptr){
    ekf_ptr_ = std::make_shared<Ekf>();
    para_ptr_->get_para("prediction/omega_max", omega_max_);
    para_ptr_->get_para("prediction/omega_filter_t", omega_filter_t_);
    para_ptr_->get_para("prediction/acc_max", acc_max_);
    para_ptr_->get_para("prediction/acc_filter_t", acc_filter_t_);
    para_ptr_->get_para("EKF/Qx", ekf_ptr_->Qt(0, 0));
    para_ptr_->get_para("EKF/Qy", ekf_ptr_->Qt(1, 1));
    para_ptr_->get_para("EKF/Qz", ekf_ptr_->Qt(2, 2));
    para_ptr_->get_para("EKF/Qvx", ekf_ptr_->Qt(3, 3));
    para_ptr_->get_para("EKF/Qvy", ekf_ptr_->Qt(4, 4));
    para_ptr_->get_para("EKF/Qvz", ekf_ptr_->Qt(5, 5));
    para_ptr_->get_para("EKF/Qroll", ekf_ptr_->Qt(6, 6));
    para_ptr_->get_para("EKF/Qpitch", ekf_ptr_->Qt(7, 7));
    para_ptr_->get_para("EKF/Qyaw", ekf_ptr_->Qt(8, 8));
    
    para_ptr_->get_para("EKF/Rx", ekf_ptr_->Rt(0, 0));
    para_ptr_->get_para("EKF/Ry", ekf_ptr_->Rt(1, 1));
    para_ptr_->get_para("EKF/Rz", ekf_ptr_->Rt(2, 2));
    para_ptr_->get_para("EKF/Rvx", ekf_ptr_->Rt(3, 3));
    para_ptr_->get_para("EKF/Rvy", ekf_ptr_->Rt(4, 4));
    para_ptr_->get_para("EKF/Rvz", ekf_ptr_->Rt(5, 5));
    para_ptr_->get_para("EKF/Rroll", ekf_ptr_->Rt(6, 6));
    para_ptr_->get_para("EKF/Rpitch", ekf_ptr_->Rt(7, 7));
    para_ptr_->get_para("EKF/Ryaw", ekf_ptr_->Rt(8, 8));


    para_ptr_->get_para("EKF/filtered_v_k_track", filtered_v_k_track_);
    para_ptr_->get_para("EKF/filtered_v_k_land", filtered_v_k_land_);
    para_ptr_->get_para("EKF/filtered_q_k", filtered_q_k_);
    para_ptr_->get_para("EKF/filtered_q_diff_k", filtered_q_diff_k_);

    filtered_v_.setZero();
    filtered_q_ = Eigen::Quaterniond(1, 0, 0, 0);
  }

};

} // namespace ekf_server
