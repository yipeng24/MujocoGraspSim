#include "node/plan_node.h"


Planner::Planner(std::shared_ptr<ShareDataManager> dataManagerPtr,
                 std::shared_ptr<parameter_server::ParaeterSerer> paraPtr,
                 std::shared_ptr<fast_planner::SDFMap> sdf_map,
                 std::shared_ptr<vis_interface::VisInterface> visPtr,
                 std::shared_ptr<clutter_hand::CH_RC_SDF> rc_sdf_ptr):
                 dataManagerPtr_(dataManagerPtr),
                 paraPtr_(paraPtr),
                 sdf_map_(sdf_map),
                 visPtr_(visPtr)
{
  paraPtr_->get_para("plan_hz", plan_hz_);
  paraPtr_->get_para("use_map", use_map_);
  paraPtr_->get_para("plan_estimated_duration", plan_estimated_duration_);   
  paraPtr_->get_para("land_no_plan_time", land_no_plan_time_);   
  paraPtr_->get_para("tracking_height_expect", tracking_height_expect_); 
  paraPtr_->get_para("tracking_dis_expect", tracking_dis_expect_); 
  paraPtr_->get_para("tracking_angle_expect", tracking_angle_expect_); 
  paraPtr_->get_para("Webvis_hz", web_vis_hz_);   
  paraPtr_->get_para("local_horizon", local_horizon_);   


  goal_method_ = sdf_map_->getGoalMethod();

  prePtr_ = std::make_shared<prediction::Predict>(paraPtr_);
  trajoptPtr_ = std::make_shared<traj_opt::TrajOpt>(paraPtr_);
  tlplannerPtr_ = std::make_shared<tlplanner::TLPlanner>(paraPtr_);
  
  gridmapPtr_ = std::make_shared<map_interface::MapInterface>(sdf_map_);
  envPtr_ = std::make_shared<env::Env>(paraPtr_, gridmapPtr_);
  prePtr_->set_gridmap_ptr(gridmapPtr_);
  trajoptPtr_->set_gridmap_ptr(gridmapPtr_);

  tlplannerPtr_->set_gridmap_ptr(gridmapPtr_);
  tlplannerPtr_->set_env_ptr(envPtr_);
  tlplannerPtr_->set_sdf_ptr(sdf_map_);
  tlplannerPtr_->set_pre_ptr(prePtr_);
  tlplannerPtr_->set_trajopt_ptr(trajoptPtr_);
  tlplannerPtr_->set_data_ptr(dataManagerPtr_);

  #ifdef RRT_STAR
  tlplannerPtr_->set_rrt_star_ptr(paraPtr_,gridmapPtr_);
  #endif

  envPtr_->set_vis_ptr(visPtr_);
  prePtr_->set_vis_ptr(visPtr_);
  trajoptPtr_->set_vis_ptr(visPtr_);
  tlplannerPtr_->set_vis_ptr(visPtr_);

  std::vector<Eigen::Vector3d> path;
  visPtr_->visualize_path(path, "astar");
  visPtr_->visualize_pointcloud(path, "astar_vp");
  visPtr_->visualize_path(path, "car_predict");
  visPtr_->visualize_pointcloud(path, "car_predict_pt");
  visPtr_->visualize_pointcloud(path, "mid_waypts");
  visPtr_->visualize_path(path, "rrt_star_final_path");
  visPtr_->visualize_pointcloud(path, "rrt_star_final_wpts");

  vector<vector<Eigen::Vector3d>> routes;
  visPtr_->visualize_path_list(routes, "rrt_star_paths");

  std::vector<std::pair<Eigen::Vector3d, double>> pcl_i;
  visPtr_->visualize_pointcloud_itensity(pcl_i, "IG_gains");

  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> lines;
  visPtr_->visualize_arrows(lines, "IG_yaws");
  visPtr_->visualize_pairline(lines, "IG_edges");

  getIGPrarm();

  INFO_MSG_GREEN("[plan node] Planner Init Done.");

}

void Planner::plan_thread(){
  while (!dataManagerPtr_->s_exit_)
  {
    int thread_dur = (int) (1000.0 / (double) plan_hz_);
    if (plan_hz_ < 0) continue;
    TimePoint t0 = TimeNow();
    PlanMode mode = get_mode();
    INFO_MSG("------------------- Plan Thread(" <<  get_mode_name() <<") -----------------------");
    #ifdef SS_DBUS
    INFO_MSG("[systime] " << TimeNow() / 1e6);
    #endif
    if (mode == IDLE){
      std::this_thread::sleep_for(std::chrono::milliseconds(thread_dur));
      continue;
    }

    
    //! 1. 获得 当前状态
    Odom odom_data;
    if (!dataManagerPtr_->get_odom(dataManagerPtr_->odom_info_, odom_data)){
      set_state(PLANFAIL);
      std::this_thread::sleep_for(std::chrono::milliseconds(thread_dur));
      continue;
    } 
    Eigen::Vector3d odom_p = odom_data.odom_p_;
    Eigen::Vector3d odom_v = odom_data.odom_v_;
    // double odom_yaw = rot_util::quaternion2yaw(odom_data.odom_q_);
    INFO_MSG("[plan node] get odom: " << odom_p.transpose());
    INFO_MSG("[plan node] get vel: " << odom_v.transpose());
    if (mode == HOVER){
      dataManagerPtr_->save_hover_p(odom_p, TimeNow(), dataManagerPtr_->traj_info_);
      set_state(NOTNEEDPLAN);
      INFO_MSG("[plan node] hovering...");
    }

    
    //! 2. 获得 目标状态
    Odom goal_data;
    bool get_new_goal = false;

    // 通过 rviz 发布目标点
    if(goal_method_ == RVIZ_GOAL)
    {
      dataManagerPtr_->get_odom(dataManagerPtr_->goal_info_, goal_data);
      last_goal_data_ = goal_data;
    }
    else{
      ROS_ERROR("GOAL METHOD ERROR");
    }

    if (mode == HOVER){
      BAG(dataManagerPtr_->BagPtr_->write_float_flatmsg("/state", HOVER));
      std::this_thread::sleep_for(std::chrono::milliseconds(thread_dur));
      continue;
    }


    //! 3 获得 地图信息
    // lock until this plan loop finish
    // ROS_WARN("try to get map");
    if (use_map_){
      if (!dataManagerPtr_->map_received_){
        ROS_WARN("not receive map");
        set_state(PLANFAIL);
        std::this_thread::sleep_for(std::chrono::milliseconds(thread_dur));
        continue;
      }
      std::lock_guard<std::mutex> map_lck(dataManagerPtr_->map_mutex_);
      // gridmapPtr_->from_msg(dataManagerPtr_->map_msg_);
      INFO_MSG("[plan node] get map");
      static int map_cnt = 0;
      if (map_cnt++ < 2){
        std::this_thread::sleep_for(std::chrono::milliseconds(thread_dur));
        continue;
      }
    }


    //! 4. 从上次执行轨迹获得初始状态
    TimePoint t00 = TimeNow();
    TimePoint replan_stamp = addDuration(t00, plan_estimated_duration_);
    double replan_t = 0.0;
    Odom init_state;
    TrajData traj_now;
    bool traj_last_valid = has_traj_last_ && 
                           (traj_last_.state_ == TrajData::D5 || traj_last_.state_ == TrajData::D7);
    if (traj_last_valid){
      replan_t = durationSecond(replan_stamp, traj_last_.start_time_);
      replan_t = replan_t <= traj_last_.getTotalDuration() ? replan_t : traj_last_.getTotalDuration() - 1e-3;
      traj_last_valid = traj_last_valid && (replan_t <= traj_last_.getTotalDuration());
    }
    // last traj not valid, plan from current odom
    if (!traj_last_valid){
    // if (!has_traj_last_){
      init_state = odom_data;
      init_state.odom_a_.setZero();
      init_state.odom_dyaw_ = 0.0;
      ROS_ERROR("last traj not valid, plan from current odom");
      std::cout << "init_p:" << init_state.odom_p_.transpose() << std::endl;
    }
    // plan from last traj
    else{
      init_state.odom_p_ = traj_last_.getPos(replan_t);
      init_state.odom_v_ = traj_last_.getVel(replan_t);
      init_state.odom_a_ = traj_last_.getAcc(replan_t);
      init_state.odom_j_ = traj_last_.getJer(replan_t);
      init_state.odom_q_ = rot_util::yaw2quaternion(traj_last_.getAngle(replan_t));
      init_state.odom_dyaw_ = traj_last_.getAngleRate(replan_t);
      if (traj_last_.getTrajType() == TrajType::NOYAW){
        init_state.odom_q_ = odom_data.odom_q_;
        init_state.odom_dyaw_ = 0.0;
      }
    }
    INFO_MSG("[plan node] get init state");


    //! 5. planner
    tlplanner::TLPlanner::PlanResState plan_res;
    bool no_need_plan = false;
    switch (mode)
    {
    case GOAL:
      {
        GData<double> track_dis_data, track_h_data;
        dataManagerPtr_->get_data(dataManagerPtr_->tracking_dis_info_, track_dis_data);
        dataManagerPtr_->get_data(dataManagerPtr_->tracking_height_info_, track_h_data);
        tracking_dis_expect_ = track_dis_data.data_;
        tracking_height_expect_ = track_h_data.data_;

        Eigen::Vector3d goal = goal_data.odom_p_;

        goal_data.odom_p_ = goal;
        goal_data.odom_v_ = Eigen::Vector3d(0.0,0.0,0.0);

        // 通过 rviz 发布目标点
        if(goal_method_==RVIZ_GOAL)
        get_new_goal = (!has_last_goal_ || (goal_data.odom_p_ - last_goal_).norm() > 0.2);

        // replan if get new goal
        if (get_new_goal) {
          INFO_MSG("[plan node] get new goal: " << goal_data.odom_p_.transpose());
          plan_res = tlplannerPtr_->plan_goal(init_state, goal_data, traj_now);
          if (plan_res == tlplanner::TLPlanner::PlanResState::PLANSUCC){
            has_last_goal_ = true;
            last_goal_stamp_ = goal_data.odom_time_stamp_ms_;
            last_goal_ = goal_data.odom_p_;
          }
          INFO_MSG_YELLOW("traj_cur duration: " << traj_now.getTotalDuration());
        }
        // else replan if last traj get occ
        else{
          bool valid = false;
          if (has_traj_last_){
            valid = tlplannerPtr_->valid_cheack(traj_last_);
          }
          if (!valid || emergency_stop_){
            std::cout << "emergency_stop_:" << emergency_stop_ << std::endl;
            std::cout << "valid:" << valid << std::endl;
            ROS_ERROR("last traj not valid");
            goal_data = last_goal_data_;
            plan_res = tlplannerPtr_->plan_goal(init_state, goal_data, traj_now);
            if (plan_res == tlplanner::TLPlanner::PlanResState::PLANSUCC){
              has_last_goal_ = true;
              last_goal_stamp_ = goal_data.odom_time_stamp_ms_;
              last_goal_ = goal_data.odom_p_;
            }
            INFO_MSG_YELLOW("traj_cur duration: " << traj_now.getTotalDuration());
          }else{
            // 没有新目标，也没有occ，直接执行上一条轨迹
            no_need_plan = true;
          }
        }
        break;
      }
    default:
      break;
    }

    bool plan_time_valid = true;
    double plan_duration = durationSecond(TimeNow(), t00);
    if (plan_duration > plan_estimated_duration_){
      INFO_MSG_RED("[plan node] Plan Time Exceed: " << plan_duration);
      plan_time_valid = false;
      INFO_MSG_RED("[plan node] Plan Time Exceed Cause Replan Fail.");
    }
    INFO_MSG("[plan node] plan done");

    //! 6. check
    // if no need to replan, directly return
    if (no_need_plan){
      INFO_MSG_GREEN("[plan node] NO NEED TO REPLAN.");
      set_state(NOTNEEDPLAN);
    }
    // if get the new plan result, check it and publish it
    else{
      bool valid = false;
      if (plan_res == tlplanner::TLPlanner::PLANSUCC){
        valid = tlplannerPtr_->valid_cheack(traj_now);
        // if (!plan_time_valid){
        //   valid = false;
        // }
      }
      if (valid){
        traj_now.start_time_ = replan_stamp;
        dataManagerPtr_->write_traj(traj_now, dataManagerPtr_->traj_info_);
        traj_last_ = traj_now;
        has_traj_last_ = true;
        INFO_MSG_GREEN("[plan node] REPLAN SUCCESS.");
        emergency_stop_ = false;
        set_state(PLANSUCC);
      }else if (has_traj_last_ && !tlplannerPtr_->valid_cheack(traj_last_)){
        dataManagerPtr_->save_hover_p(odom_p, TimeNow(), dataManagerPtr_->traj_info_);
        INFO_MSG_RED("[plan node] EMERGENCY STOP!");
        emergency_stop_ = true;
        set_state(PLANFAIL);
      }else{
        INFO_MSG_RED("[plan node] REPLAN FAIL, EXECUTE LAST TRAJ!");
        set_state(PLANFAIL);
      }
    }

    TimePoint t1 = TimeNow();
    double d0 = durationSecond(t1, t0) * 1e3; // ms
    if (d0 > thread_dur){
      INFO_MSG_RED("[plan node] Error! thread time exceed! " << d0 << "ms");
      continue;
    }else{
      int tr = floor(thread_dur - d0);
      std::this_thread::sleep_for(std::chrono::milliseconds(tr));
    }
  }
  INFO_MSG_RED("[plan node] Thread Exit.");
   
}