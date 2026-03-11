#ifndef DATA_MANAGER_HPP
#define DATA_MANAGER_HPP
#include <atomic>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <deque>
#include <concurrentqueue/concurrentqueue.h>

#include "util_gym/util_gym.hpp"
#include "traj_opt/trajectory.hpp"
#include "target_ekf/target_ekf_server.hpp"
#include "parameter_server/parameter_server.hpp"

#ifdef ROS
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <quadrotor_msgs/OccMap3d.h>
#include <sensor_msgs/Joy.h>
#endif

template <class DataType>
struct GData{
    DataType data_;
    TimePoint time_stamp_us_;
};

template <class DataType>
struct GInfo{
    std::mutex mutex_;
    std::atomic_bool data_received_ = ATOMIC_VAR_INIT(false);
    GData<DataType> data_;
};

template <class DataType>
class GInfoVec{
 public:
    std::mutex mutex_;
    std::atomic_bool data_received_ = ATOMIC_VAR_INIT(false);
    std::deque<GData<DataType>> data_vec_;
    const int vec_length_ = 50;
};

class PoseStamp{
 public:
    Eigen::Vector3d t_;
    Eigen::Quaterniond q_;
    PoseStamp(){
        t_.setZero();
        q_ = Eigen::Quaterniond(1, 0, 0, 0);
    }
};

class OdomBase
{
 public:
    Eigen::Vector3d odom_p_;
    Eigen::Quaterniond odom_q_;
    double odom_dyaw_;
    TimePoint odom_time_stamp_ms_;

    OdomBase(){
        odom_p_.setZero();
        odom_q_ = Eigen::Quaterniond(1, 0, 0, 0);
        odom_dyaw_ = 0.0;
    }
};

class Odom:public OdomBase{
 public:
    Eigen::Vector3d odom_v_;
    Eigen::Vector3d odom_a_;
    Eigen::Vector3d odom_j_;
    Eigen::Vector3d theta_;
    Eigen::Vector3d dtheta_;
    Eigen::Vector3d end_p_;
    Eigen::Quaterniond end_q_;

    Odom(){
        odom_v_.setZero();
        odom_a_.setZero();
        odom_j_.setZero();
        theta_.setZero();
        dtheta_.setZero();
    }
};

class CarOdom:public OdomBase{
 public:
    double odom_v_;
    double odom_a_;

    CarOdom(){
        odom_v_ = 0.0;
        odom_a_ = 0.0;
    }
};

const int odom_vec_length_ = 100;

template <class Data>
struct OdomInfo
{
    std::mutex mutex_;
    std::atomic_bool odom_received_ = ATOMIC_VAR_INIT(false);
    std::deque<Data> data_vec_;
};


class TrajData
{
public:
    enum TrajState{
        Hover,
        D5,
        D7,
        EndLanding,
    };
public:
    TrajState state_;
    int traj_id_;
    TimePoint start_time_;
    Trajectory<5> traj_d5_;
    Trajectory<7> traj_d7_;
    Eigen::Vector3d hover_p_;
    double yaw_;

public:
    inline double getTotalDuration() const{
        double T = 0.0;
        switch (state_)
        {
        case D5:
            T = traj_d5_.getTotalDuration();
            break;
        case D7:
            T = traj_d7_.getTotalDuration();
            break;
        default:
            break;
        }
        return T;
    }
    inline Eigen::Vector3d getPos(const double& t) const{
        Eigen::Vector3d pos(0.0, 0.0, 0.0);
        switch (state_)
        {
        case D5:
            pos = traj_d5_.getPos(t);
            break;
        case D7:
            pos = traj_d7_.getPos(t);
            break;
        default:
            break;
        }
        return pos;
    }
    inline Eigen::Vector3d getVel(const double& t) const{
        Eigen::Vector3d vel(0.0, 0.0, 0.0);
        switch (state_)
        {
        case D5:
            vel = traj_d5_.getVel(t);
            break;
        case D7:
            vel = traj_d7_.getVel(t);
            break;
        default:
            break;
        }
        return vel;
    }
    inline Eigen::Vector3d getAcc(const double& t) const{
        Eigen::Vector3d acc(0.0, 0.0, 0.0);
        switch (state_)
        {
        case D5:
            acc = traj_d5_.getAcc(t);
            break;
        case D7:
            acc = traj_d7_.getAcc(t);
            break;
        default:
            break;
        }
        return acc;
    }
    inline Eigen::Vector3d getJer(const double& t) const{
        Eigen::Vector3d jer(0.0, 0.0, 0.0);
        switch (state_)
        {
        case D5:
            jer = traj_d5_.getJer(t);
            break;
        case D7:
            jer = traj_d7_.getJer(t);
            break;
        default:
            break;
        }
        return jer;
    }
    inline Eigen::Vector3d getSna(const double& t) const{
        Eigen::Vector3d sna(0.0, 0.0, 0.0);
        switch (state_)
        {
        case D5:
            sna = traj_d5_.getSna(t);
            break;
        case D7:
            sna = traj_d7_.getSna(t);
            break;
        default:
            break;
        }
        return sna;
    }
    inline Eigen::Vector3d getCra(const double& t) const{
        Eigen::Vector3d cra(0.0, 0.0, 0.0);
        switch (state_)
        {
        case D5:
            cra = traj_d5_.getCra(t);
            break;
        case D7:
            cra = traj_d7_.getCra(t);
            break;
        default:
            break;
        }
        return cra;
    }
    inline double getAngle(const double& t) const{
        double angle = 0.0;
        switch (state_)
        {
        case D5:
            angle = traj_d5_.getAngle(t);
            break;
        case D7:
            angle = traj_d7_.getAngle(t);
            break;
        default:
            break;
        }
        return angle;
    }
    inline double getAngleRate(const double& t) const{
        double angler = 0.0;
        switch (state_)
        {
        case D5:
            angler = traj_d5_.getAngleRate(t);
            break;
        case D7:
            angler = traj_d7_.getAngleRate(t);
            break;
        default:
            break;
        }
        return angler;
    }
    inline Eigen::VectorXd getTheta(const double& t) const{
        Eigen::VectorXd thetas;
        switch (state_)
        {
        case D5:
            thetas = traj_d5_.getTheta(t);
            break;
        case D7:
            thetas = traj_d7_.getTheta(t);
            break;
        default:
            break;
        }
        return thetas;
    }
    inline Eigen::VectorXd getThetaRate(const double& t) const{
        Eigen::VectorXd d_thetas;
        switch (state_)
        {
        case D5:
            d_thetas = traj_d5_.getThetaRate(t);
            break;
        case D7:
            d_thetas = traj_d7_.getThetaRate(t);
            break;
        default:
            break;
        }
        return d_thetas;
    }
    inline TrajType getTrajType() const{
        TrajType type = NONE;
        switch (state_)
        {
        case D5:
            type = traj_d5_.getType();
            break;
        case D7:
            type = traj_d7_.getType();
            break;
        default:
            break;
        }
        return type;
    }
    inline TrajState getTrajState() const{
        return state_;
    }
};
struct TrajInfo
{
    std::mutex mutex_;
    std::atomic_bool traj_received_ = ATOMIC_VAR_INIT(false);
    TrajData data_;
};

#ifdef SS_DBUS
struct ImgData{
    std::string name;
    int id;
    cv::Mat img_;
    TimePoint time_stamp_us_;
};
#endif


class ShareDataManager{
public:
    OdomInfo<Odom> odom_info_;
    OdomInfo<Odom> goal_info_;
    OdomInfo<CarOdom> car_info_;
    TrajInfo traj_info_;

    GInfo<double> tracking_dis_info_;
    GInfo<double> tracking_angle_info_;
    GInfo<double> tracking_height_info_;

#ifdef SS_DBUS
    moodycamel::ConcurrentQueue<ImgData> img_front_queue_;
    moodycamel::ConcurrentQueue<ImgData> img_down_queue_;
#endif

    // Trigger
    std::atomic_bool s_exit_ = ATOMIC_VAR_INIT(false);
    std::atomic_bool auto_mode_ = ATOMIC_VAR_INIT(false);
    std::atomic_bool force_manual_mode_ = ATOMIC_VAR_INIT(false);
    std::atomic_bool plan_trigger_received_ = ATOMIC_VAR_INIT(false);
    std::atomic_bool land_trigger_received_ = ATOMIC_VAR_INIT(false);
    std::atomic_bool map_received_ = ATOMIC_VAR_INIT(false);

    // ekf attachment
    std::shared_ptr<ekf_server::EKFServer> car_ekf_ptr_;

    /*********** Ros Variables ***************/
    #ifdef ROS
    quadrotor_msgs::OccMap3d map_msg_;
    std::mutex map_mutex_;
    #endif
    /****************************************/


public:
    ShareDataManager(std::shared_ptr<parameter_server::ParaeterSerer>& paraPtr){
        car_ekf_ptr_ = std::make_shared<ekf_server::EKFServer>(paraPtr);
        paraPtr->get_para("tracking_height_expect", tracking_height_info_.data_.data_); 
        paraPtr->get_para("tracking_dis_expect", tracking_dis_info_.data_.data_); 
        paraPtr->get_para("tracking_angle_expect", tracking_angle_info_.data_.data_); 
        tracking_height_info_.data_received_ = true;
        tracking_dis_info_.data_received_ = true;
        tracking_angle_info_.data_received_ = true;

        #ifdef SS_DBUS
        
        #endif
    }
    ~ShareDataManager(){}

    template <class DataType>
    bool get_data(GInfo<DataType>& g_info, GData<DataType>& g_data){
        if (!g_info.data_received_) return false;
        std::lock_guard<std::mutex> lck(g_info.mutex_);
        g_data = g_info.data_;
        return true;
    }

    template <class DataType>
    void write_data(const GData<DataType>& g_data, GInfo<DataType>& g_info){
        std::lock_guard<std::mutex> lck(g_info.mutex_);
        g_info.data_ = g_data;
        g_info.data_received_ = true;
    }

    template <class DataType>
    bool get_data(GInfoVec<DataType>& g_info, GData<DataType>& g_data){
        if (!g_info.data_received_) return false;
        std::lock_guard<std::mutex> lck(g_info.mutex_);
        g_data = g_info.data_vec_.back();
        return true;
    }

    template <class DataType>
    void write_data(const GData<DataType>& g_data, GInfoVec<DataType>& g_info){
        std::lock_guard<std::mutex> lck(g_info.mutex_);
        g_info.data_vec_.push_back(g_data);
        if (g_info.data_vec_.size() > g_info.vec_length_) g_info.data_vec_.pop_front();
        g_info.data_received_ = true;
    }

    template <class Data>
    bool get_odom(OdomInfo<Data>& odom_info, Data& odom_data){
        if (!odom_info.odom_received_) return false;
        std::lock_guard<std::mutex> lck(odom_info.mutex_);
        odom_data = odom_info.data_vec_.back();
        return true;
    }

    template <class Data>
    void write_odom(const Data& odom_data, OdomInfo<Data>& odom_info){
        std::lock_guard<std::mutex> lck(odom_info.mutex_);
        odom_info.data_vec_.push_back(odom_data);
        if (odom_info.data_vec_.size() > odom_vec_length_) odom_info.data_vec_.pop_front();
        odom_info.odom_received_ = true;
    }

    bool get_traj(TrajInfo& traj_info, TrajData& traj_data){
        if (!traj_info.traj_received_) return false;
        std::lock_guard<std::mutex> lck(traj_info.mutex_);
        traj_data = traj_info.data_;
        return true;
    }

    // only fill data_.traj_id_, other data should be filled
    void write_traj(const TrajData& traj_data, TrajInfo& traj_info){
        std::lock_guard<std::mutex> lck(traj_info.mutex_);
        if (!traj_info.traj_received_) traj_info.data_.traj_id_ = 0;
        int old_id = traj_info.data_.traj_id_;
        traj_info.data_ = traj_data;
        traj_info.data_.traj_id_ = old_id + 1;
        traj_info.traj_received_ = true;
    }

    void save_hover_p(const Eigen::Vector3d& hover_p, const TimePoint& stamp, TrajInfo& traj_info){
        std::lock_guard<std::mutex> lck(traj_info.mutex_);
        if (!traj_info.traj_received_) traj_info.data_.traj_id_ = 0;
        traj_info.data_.state_ = TrajData::Hover;
        traj_info.data_.hover_p_ = hover_p;
        traj_info.data_.start_time_ = stamp;
        traj_info.data_.traj_id_++;
        traj_info.traj_received_ = true;
    }

    void save_end_landing(const TimePoint& stamp, TrajInfo& traj_info){
        std::lock_guard<std::mutex> lck(traj_info.mutex_);
        if (!traj_info.traj_received_) traj_info.data_.traj_id_ = 0;
        traj_info.data_.state_ = TrajData::EndLanding;
        traj_info.data_.start_time_ = stamp;
        traj_info.data_.traj_id_++;
        traj_info.traj_received_ = true;
    }

    // bool get_car_odom(CarOdom& odom_data){
    //     if (!car_ekf_ptr_->update_data_received()) return false;
    //     Eigen::Vector3d p_pre;
    //     double v_pre, theta_pre, omega_pre;
    //     car_ekf_ptr_->get_predict_state(p_pre, v_pre, theta_pre, omega_pre);
    //     odom_data.odom_p_ = p_pre;
    //     odom_data.odom_v_ = v_pre;
    //     odom_data.odom_a_ = 0.0;
    //     odom_data.odom_q_ = rotation_util::RotUtil::yaw2quaternion(theta_pre);
    //     odom_data.odom_dyaw_ = omega_pre;
    //     odom_data.odom_time_stamp_ms_ = TimeNow();
    //     return true;
    // }

    bool get_car_odom(Odom& odom_data){
        if (!car_ekf_ptr_->update_data_received()) return false;
        Eigen::Vector3d p_pre, v_pre, a_pre, rpy_pre;
        double omega_pre;
        car_ekf_ptr_->get_predict_state(p_pre, v_pre, a_pre, rpy_pre, omega_pre);
        odom_data.odom_p_ = p_pre;
        odom_data.odom_v_ = v_pre;
        odom_data.odom_a_ = a_pre;
        odom_data.odom_q_ = rotation_util::RotUtil::euler2quaternion(rpy_pre);
        odom_data.odom_dyaw_ = omega_pre;
        odom_data.odom_time_stamp_ms_ = TimeNow();
        return true;
    }

public:
    #ifdef SS_DBUS
    /*********** SS_DBUS Variables ***************/

    /****************************************/
    #endif
};
#endif