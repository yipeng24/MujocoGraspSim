#include <node/traj_server_node.h>

using rot_util = rotation_util::RotUtil;

TrajServer::TrajServer(std::shared_ptr<ShareDataManager> dataManagerPtr,
                       std::shared_ptr<parameter_server::ParaeterSerer> paraPtr):
                       dataManagerPtr_(dataManagerPtr),
                       paraPtr_(paraPtr)
{
    paraPtr_->get_para("PD_Ctrl/Kp_horiz", Kp_horiz_);
    paraPtr_->get_para("PD_Ctrl/Kd_horiz", Kd_horiz_);
    paraPtr_->get_para("PD_Ctrl/Kp_vert", Kp_vert_);
    paraPtr_->get_para("PD_Ctrl/Kd_vert", Kd_vert_);
    paraPtr_->get_para("TrajServer/dyaw_max", dyaw_max_);
    paraPtr_->get_para("grasp_delay", t_delay_);
    paraPtr_->get_para("grasp_mode", grasp_mode_);
    last_t_theta_cmd_sent_ = ros::Time::now();
}

bool TrajServer::exe_traj(const Odom& odom_data, const TrajData& traj_data){
    TimePoint sample_time = addDuration(TimeNow(), dt_to_future_s_);
    double t = durationSecond(sample_time, traj_data.start_time_);

// 1. 新轨迹检测与状态重置
    if (traj_data.traj_id_ != last_traj_id_) {
        last_traj_id_ = traj_data.traj_id_;
        run_zero_sent_ = false;   // 重置起始信号标志
        grab_triggered_ = false;  // 重置抓取信号标志
        reached_end_ = false;     // 重置终点标记
    }

    Eigen::Vector3d odom_p_raw = odom_data.odom_p_;
    Eigen::Vector3d odom_v = odom_data.odom_v_;
    Eigen::Vector3d odom_a = odom_data.odom_a_;
    double odom_yaw = rot_util::quaternion2yaw(odom_data.odom_q_);
    TimePoint odom_stamp_ms = odom_data.odom_time_stamp_ms_;

    if (t > 0){
        int current_grasp_cmd = 2; // 默认所有时刻都发送 2 (不操作)

        if (traj_data.state_ == TrajData::EndLanding){
            has_stop_propeller_ = stop_propeller();
            return true;
        }
        if (traj_data.state_ == TrajData::Hover){
            Eigen::Vector3d zero;
            zero.setZero();
            publish_cmd(traj_data.traj_id_, traj_data.hover_p_, zero, zero, zero,
                        2,
                        last_yaw_, 0, false, odom_p_raw, odom_v, odom_a, odom_yaw,
                        odom_stamp_ms, sample_time);
            return true;
        }
        if (t > traj_data.getTotalDuration()){
            if (!reached_end_) {
                reached_end_ = true;
                end_reach_time_ = TimeNow();
            }


            // 到达延时且未发送过 1 时，触发一次抓取
            if (!grab_triggered_ && durationSecond(TimeNow(), end_reach_time_) >= t_delay_) {
                current_grasp_cmd = grasp_mode_ ? 1 : 0;
                grab_triggered_ = true; // 锁定，后续变回 2
            }


            // INFO_MSG_RED("[traj_server] Traj execute done!  Hovering...");
            Eigen::Vector3d p, zero;
            zero.setZero();
            p = traj_data.getPos(traj_data.getTotalDuration());
            
            // if ((traj_data.getTrajType() == TrajType::WITHYAWANDTHETA) & (t < 1e-3 + traj_data.getTotalDuration())){
            if (traj_data.getTrajType() == TrajType::WITHYAWANDTHETA){
                Eigen::VectorXd theta, dtheta;
                theta = traj_data.getTheta(traj_data.getTotalDuration());
                dtheta = traj_data.getThetaRate(traj_data.getTotalDuration());

                publish_cmd(traj_data.traj_id_, p, zero, zero, zero, theta, dtheta, current_grasp_cmd,
                            last_yaw_, 0, false, odom_p_raw, odom_v, odom_a, odom_yaw,
                            odom_stamp_ms, sample_time);

            }
            else{

                publish_cmd(traj_data.traj_id_, p, zero, zero, zero,
                        current_grasp_cmd,
                        last_yaw_, 0, false, odom_p_raw, odom_v, odom_a, odom_yaw,
                        odom_stamp_ms, sample_time);
            }

            return true;
        }
        Eigen::Vector3d p, v, a, j, s, c;
        double yaw, dyaw;
        p = traj_data.getPos(t);
        v = traj_data.getVel(t);
        a = traj_data.getAcc(t);
        j = traj_data.getJer(t);
        s = traj_data.getSna(t);
        yaw = traj_data.getAngle(t);
        dyaw = traj_data.getAngleRate(t);

        if((p-last_pos_cmd_).norm() > 1.0 ){
            INFO_MSG_RED("[exe_traj] Error! pos jump, please adjust the plan_hz and plan_estimated_time.");
            INFO_MSG_RED("last_pos_cmd_:" << last_pos_cmd_.transpose() << " p:" << p.transpose());
        }
        last_pos_cmd_ = p;
        last_vel_cmd_ = v;
        last_acc_cmd_ = a;

        // 如果本条轨迹还没发过起始信号 0，则发送一次
        if (!run_zero_sent_) {
            current_grasp_cmd = grasp_mode_ ? 0 : 1;
            run_zero_sent_ = true; // 锁定，后续运行期间变回 2
        } else {
            current_grasp_cmd = 2; // 运行期间的其余时刻发送 2
        }

        if (traj_data.getTrajType() == TrajType::NOYAW){
            double last_yaw = last_yaw_;
            publish_cmd(traj_data.traj_id_, p, v, a, j,
                        current_grasp_cmd,
                        last_yaw, 0, true, odom_p_raw, odom_v, odom_a, odom_yaw,
                        odom_stamp_ms, sample_time);
            last_yaw_ = last_yaw;
        }else{
            Eigen::Vector3d zero;
            zero.setZero();
            // publish_cmd(traj_data.traj_id_, p, v, a, j,
                        // yaw, dyaw, false, odom_p_raw, odom_v, odom_a, odom_yaw,
                        // odom_stamp_ms, sample_time);
            // last_yaw_ = yaw;
            
            // if (traj_data.getTrajType() == TrajType::WITHYAWANDTHETA){
            //     Eigen::VectorXd theta, dtheta;
            //     theta = traj_data.getTheta(t);
            //     dtheta = traj_data.getThetaRate(t);

            //     if((ros::Time::now() - last_t_theta_cmd_sent_).toSec() > 0.1){
            //         publish_cmd_theta(theta, dtheta, sample_time);
            //         last_t_theta_cmd_sent_ = ros::Time::now();
            //     }
            // }

            last_yaw_ = yaw;
            
            if (traj_data.getTrajType() == TrajType::WITHYAWANDTHETA){
                Eigen::VectorXd theta, dtheta;
                theta = traj_data.getTheta(t);
                dtheta = traj_data.getThetaRate(t);
                publish_cmd(traj_data.traj_id_, p, v, a, j, theta, dtheta, current_grasp_cmd,
                            yaw, dyaw, false, odom_p_raw, odom_v, odom_a, odom_yaw,
                            odom_stamp_ms, sample_time);
            }else{
                publish_cmd(traj_data.traj_id_, p, v, a, j,
                            current_grasp_cmd,
                            yaw, dyaw, false, odom_p_raw, odom_v, odom_a, odom_yaw,
                            odom_stamp_ms, sample_time);
            }

        }
        return true;
    }
    return false;
}

void TrajServer::cmd_thread(){
  while (!dataManagerPtr_->s_exit_)
  {    
    int thread_dur = (int) (1000.0 / (double) cmd_hz_);
    
    TimePoint t0 = TimeNow();
    //! 1. obtain odom
    Odom odom_data;
    if (!dataManagerPtr_->get_odom(dataManagerPtr_->odom_info_, odom_data)){
        std::this_thread::sleep_for(std::chrono::milliseconds(thread_dur));
        continue;
    }

    //! 2. obtain traj
    TrajData traj_data;
    if (!dataManagerPtr_->get_traj(dataManagerPtr_->traj_info_, traj_data)){
        last_yaw_ = rot_util::quaternion2yaw(odom_data.odom_q_);
        std::this_thread::sleep_for(std::chrono::milliseconds(thread_dur));
        continue;
    }

    //! 3. execute traj
    if (exe_traj(odom_data, traj_data)){
        // vis traj
        if (traj_data.traj_id_ > traj_data_last_.traj_id_){
            traj2vis(traj_data);
        }
        traj_data_last_ = traj_data;
        has_last_traj_ = true;
    }else if (has_last_traj_){
        if (traj_data.traj_id_ - traj_data_last_.traj_id_ > 1){
            INFO_MSG_RED("[traj_server] Error! trajid jump, please adjust the plan_hz and plan_estimated_time.");
        }
        exe_traj(odom_data, traj_data_last_);
    }

    TimePoint t1 = TimeNow();
    double d0 = durationSecond(t1, t0) * 1e3;
    if (d0 > thread_dur){
        INFO_MSG_RED("[traj_server] Error! thread time exceed " << d0 << "ms!");
        continue;
    }
    else{
        int tr_nano = floor(thread_dur*1e6 - d0*1e6);
        std::this_thread::sleep_for(std::chrono::nanoseconds(tr_nano));
    }
  }
  INFO_MSG_RED("[traj_server] Thread Exit.");
}

Eigen::Quaterniond TrajServer::cmd2odom(const Eigen::Vector3d& acc, const double& yaw){
    Eigen::Vector3d alpha = Eigen::Vector3d(acc.x(), acc.y(), acc.z()) + 9.8*Eigen::Vector3d(0,0,1);
    Eigen::Vector3d xC(cos(yaw), sin(yaw), 0);
    Eigen::Vector3d yC(-sin(yaw), cos(yaw), 0);
    Eigen::Vector3d xB = (yC.cross(alpha)).normalized();
    Eigen::Vector3d yB = (alpha.cross(xB)).normalized();
    Eigen::Vector3d zB = xB.cross(yB);
    Eigen::Matrix3d R;
    R.col(0) = xB;
    R.col(1) = yB;
    R.col(2) = zB;
    Eigen::Quaterniond q(R);
    return q;
}
