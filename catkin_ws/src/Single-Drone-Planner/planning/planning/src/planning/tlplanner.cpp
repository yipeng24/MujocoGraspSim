#include "planning/tlplanner.h"
using rot_util = rotation_util::RotUtil;

namespace tlplanner{
TLPlanner::TLPlanner(std::shared_ptr<parameter_server::ParaeterSerer>& para_ptr):paraPtr_(para_ptr){
    paraPtr_->get_para("Webvis_hz", web_vis_hz_);   
    paraPtr_->get_para("is_use_viewpoint", is_use_viewpoint_);   
    paraPtr_->get_para("plan_estimated_duration", plan_estimated_duration_);   

    paraPtr_->get_para("plan_horizon_len", plan_horizon_len_); 
}


TLPlanner::PlanResState TLPlanner::plan_goal(const Odom& init_state_in, const Odom& target_data, TrajData& traj_data){
    //! 3. get inital state
    Eigen::MatrixXd init_state, init_yaw, init_thetas;
    init_state.setZero(3, 4);
    init_yaw.setZero(1, 2);
    init_thetas.setZero(init_state_in.theta_.size(),2);
    init_state.col(0) = init_state_in.odom_p_;
    init_state.col(1) = init_state_in.odom_v_;
    init_state.col(2) = init_state_in.odom_a_;
    init_yaw(0, 0)    = rot_util::quaternion2yaw(init_state_in.odom_q_);
    init_yaw(0, 1)    = init_state_in.odom_dyaw_;
    init_thetas.col(0) = init_state_in.theta_;
    init_thetas.col(1) = init_state_in.dtheta_;

    INFO_MSG("init p: " << init_state.col(0).transpose());
    INFO_MSG("init v: " << init_state.col(1).transpose());
    INFO_MSG("init a: " << init_state.col(2).transpose());
    INFO_MSG("init j: " << init_state.col(3).transpose());
    INFO_MSG("init yaw: " << init_yaw);
    INFO_MSG("init thetas: " << init_thetas.col(0).transpose());
    INFO_MSG("init d_thetas: " << init_thetas.col(1).transpose());
    INFO_MSG("tar p: " << target_data.odom_p_.transpose());
    INFO_MSG("tar v: " << target_data.odom_v_.transpose());
    INFO_MSG("tar a: " << target_data.odom_a_.transpose());
    INFO_MSG("tar thetas:" << target_data.theta_.transpose());
    INFO_MSG("tar d_thetas:" << target_data.dtheta_.transpose());
    // INFO_MSG("tar theta: " << rot_util::quaternion2yaw(target_data.odom_q_));

    //! 4. path search
    Eigen::Vector3d p_start = init_state.col(0);
    std::vector<Eigen::Vector3d> path3d, way_pts;
    std::vector<Eigen::VectorXd> pathXd;
    bool generate_new_traj_success;

    std::vector<Eigen::Vector3d> vis_goal_p;
    vis_goal_p.push_back(target_data.odom_p_);
    visPtr_->visualize_pointcloud(vis_goal_p, "front_end_goal");

    INFO_MSG_GREEN_BG("----- Start search path");
    generate_new_traj_success = envPtr_->short_astar(p_start, target_data.odom_p_, path3d); // get a path3d from current pose to target pose
    if (path3d.empty() || (path3d.front() - p_start).norm() > 1e-6)
        path3d.insert(path3d.begin(), p_start);
    if (path3d.empty() || (path3d.back() - target_data.odom_p_).norm() > 1e-6)
        path3d.push_back(target_data.odom_p_);
    if(generate_new_traj_success)
        visPtr_->visualize_path(path3d, "astar");
    else{
        INFO_MSG_RED("[tlplanner] search path3d fail!");
        return FAIL;    
    }

    //! 5. traj opt
    Trajectory<7> traj_goal;
    Eigen::MatrixXd final_state, final_yaw, final_thetas;
    final_state.setZero(3, 4);
    final_yaw.setZero(1, 2);
    final_thetas.setZero(target_data.theta_.size(),2);
    final_state.col(0) = path3d.back();
    final_state.col(1) = target_data.odom_v_;
    final_yaw(0, 0) = rot_util::quaternion2yaw(target_data.odom_q_);
    final_thetas.col(0) = target_data.theta_;
    // final_thetas.col(1) = target_data.dtheta_;
    INFO_MSG("init p: " << init_state.col(0).transpose());
    INFO_MSG("end p: " << final_state.col(0).transpose());
    INFO_MSG("init yaw: " << init_yaw(0, 0));
    INFO_MSG("end yaw: " << final_yaw(0, 0));

    // pass ring pose to trajopt if enabled
    if (en_through_ring_ && trajoptPtr_){
        trajoptPtr_->set_en_through_ring(true);
        if (has_ring_pose_){
            trajoptPtr_->set_ring_pose(ring_pose_);
        }
    }else if(trajoptPtr_){
        trajoptPtr_->set_en_through_ring(false);
    }

    // 3D
    generate_new_traj_success = trajoptPtr_->generate_traj_clutter(init_state, final_state, init_yaw, 
                                                                    final_yaw, init_thetas, final_thetas,
                                                                    2.0, path3d, traj_goal); 
    
    // XD
    // generate_new_traj_success = trajoptPtr_->generate_traj_clutter(init_state, final_state, init_yaw, 
    //                                                                 final_yaw, init_thetas, final_thetas,
    //                                                                 2.0, pathXd, traj_goal); 

    auto publish_full_state_seq = [&](const Trajectory<7>& traj,
                                      const std::string& topic,
                                      const visualization_rc_sdf::Color& color){
        if (!rc_sdf_ptr_ || !trajoptPtr_) return;
        std::vector<Eigen::Vector3d> pos_seq;
        std::vector<Eigen::Quaterniond> q_seq;
        std::vector<Eigen::VectorXd> arm_seq;
        trajoptPtr_->sample_traj_states(traj, pos_seq, q_seq, arm_seq);

        visualization_msgs::MarkerArray marker_array;
        for(size_t i = 0; i < pos_seq.size(); ++i){
            rc_sdf_ptr_->getRobotMarkerArray(pos_seq[i], q_seq[i], arm_seq[i],
                                             marker_array, color, 0.25);
        }
        rc_sdf_ptr_->robotMarkersPub(marker_array, topic);
    };

    // TODO add vis seq
    if (!generate_new_traj_success) {
        INFO_MSG_RED("[tlplanner] Traj opt fail!");
        visPtr_->visualize_traj(traj_goal, "traj_failed");
        publish_full_state_seq(traj_goal, "traj_failed_robot_markers",
                               visualization_rc_sdf::Color::red);
        return FAIL;
    }
    traj_data.traj_d7_ = traj_goal;
    traj_data.state_ = TrajData::D7;

    if(web_vis_hz_ > 0){
        visPtr_->visualize_traj(traj_goal, "traj");
        publish_full_state_seq(traj_goal, "traj_robot_markers",
                               visualization_rc_sdf::Color::green);
    }

    return PLANSUCC;
}



bool TLPlanner::valid_cheack(const TrajData& traj_data, const TimePoint& cur_t){

    double t0 = durationSecond(cur_t, traj_data.start_time_); //返回轨迹开始时间到当前时间的时间差
    t0 = t0 > 0.0 ? t0 : 0.0;
    INFO_MSG("------------valid check");
    for (double t = t0; t < traj_data.getTotalDuration(); t += 0.01){ //遍历轨迹            
        // #ifndef USE_RC_SDF
        //     Eigen::Vector3d p = traj_data.getPos(t);
        //     if (gridmapPtr_->isOccupied(p)){
        //         INFO_MSG_RED("traj is invalid, hit obstacle");
        //         return false;
        //     }
        // #else
        //     if(trajoptPtr_->isOccupied_se3(traj_data,t)){
        //         INFO_MSG_RED("traj is invalid, hit obstacle");
        //         return false;
        //     }
        // #endif

        Eigen::Vector3d p = traj_data.getPos(t);
        if (gridmapPtr_->isOccupied(p)){
            INFO_MSG_RED("traj is invalid, hit obstacle");
            return false;
        }

    }

    return true;
}


void TLPlanner::cal_local_goal_from_path(std::vector<Eigen::Vector3d>& path){

    // 在外部函数中定义内部函数
    std::vector<Eigen::Vector3d> path_temp;

    Eigen::Vector3d p_a, p_b, p_o;
    for(int i = 0; i < path.size(); i++){
        if(i == 0){
            path_temp.push_back(path[i]);
        }else{
            if((path[i] - path_temp.back()).norm() < plan_horizon_len_){
                path_temp.push_back(path[i]);
            }
            else {
                p_a = path_temp.back();
                p_b = path[i];

                p_o = path[0];


                // 计算ab线段上距离 path[0] 长度为 plan_horizon_len_ 的点
                double cos_a = (p_o - p_a).dot(p_b - p_a) / ((p_o - p_a).norm() * (p_b - p_a).norm());
                // 垂线长度

                double a = 1;
                double b = -2 * cos_a * (p_o - p_a).norm();
                double c = (p_o - p_a).norm() * (p_o - p_a).norm() 
                        - plan_horizon_len_ * plan_horizon_len_;
                
                double t = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);

                path_temp.push_back(p_a + t * (p_b - p_a));
                break;
            }
        }
    }

    path = path_temp;
}

} // namespace tlplanner
