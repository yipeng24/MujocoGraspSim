#include "node/plan_node.h"

Planner::Planner(std::shared_ptr<ShareDataManager> dataManagerPtr,
                 ros::NodeHandle& nh,
                 std::shared_ptr<parameter_server::ParaeterSerer> paraPtr,
                 std::shared_ptr<vis_interface::VisInterface> visPtr):
                 dataManagerPtr_(dataManagerPtr),
                 nh_(nh),
                 paraPtr_(paraPtr),
                 visPtr_(visPtr)
{
  paraPtr_->get_para("plan_hz", plan_hz_);
  paraPtr_->get_para("plan_estimated_duration", plan_estimated_duration_);   
  paraPtr_->get_para("local_horizon", local_horizon_);   

  paraPtr_->get_para("vmax", v_max_);

  rc_sdf_ptr_ = std::make_shared<clutter_hand::CH_RC_SDF>();
  rc_sdf_ptr_->initMap(nh, true, false);

  int goal_method;
  paraPtr_->get_para("goal_method", goal_method);
  goal_method_ = (enum GoalMethod)goal_method;

  // random odom config
  int use_random_flag = 0;
  paraPtr_->get_para("odom_random/use_random_odom", use_random_flag);
  use_random_odom_ = (use_random_flag != 0);
  paraPtr_->get_para("odom_random/pos_x_min", odom_rand_pos_min_.x());
  paraPtr_->get_para("odom_random/pos_x_max", odom_rand_pos_max_.x());
  paraPtr_->get_para("odom_random/pos_y_min", odom_rand_pos_min_.y());
  paraPtr_->get_para("odom_random/pos_y_max", odom_rand_pos_max_.y());
  paraPtr_->get_para("odom_random/pos_z_min", odom_rand_pos_min_.z());
  paraPtr_->get_para("odom_random/pos_z_max", odom_rand_pos_max_.z());
  paraPtr_->get_para("odom_random/yaw_min", odom_rand_yaw_min_);
  paraPtr_->get_para("odom_random/yaw_max", odom_rand_yaw_max_);
  paraPtr_->get_para("odom_random/theta_0_min", odom_rand_theta_min_.x());
  paraPtr_->get_para("odom_random/theta_0_max", odom_rand_theta_max_.x());
  paraPtr_->get_para("odom_random/theta_1_min", odom_rand_theta_min_.y());
  paraPtr_->get_para("odom_random/theta_1_max", odom_rand_theta_max_.y());
  paraPtr_->get_para("odom_random/theta_2_min", odom_rand_theta_min_.z());
  paraPtr_->get_para("odom_random/theta_2_max", odom_rand_theta_max_.z());

  trajoptPtr_ = std::make_shared<traj_opt::TrajOpt>(nh, paraPtr_);
  tlplannerPtr_ = std::make_shared<tlplanner::TLPlanner>(paraPtr_);
  
  gridmapPtr_ = std::make_shared<map_interface::MapInterface>(nh_);
  envPtr_ = std::make_shared<env::Env>(paraPtr_, gridmapPtr_);

  INFO_MSG_GREEN("[CH_PLAN] ptr init done");

  trajoptPtr_->set_gridmap_ptr(gridmapPtr_);
  trajoptPtr_->set_rcsdf_ptr(rc_sdf_ptr_);  

  tlplannerPtr_->set_gridmap_ptr(gridmapPtr_);
  tlplannerPtr_->set_env_ptr(envPtr_);
  // tlplannerPtr_->set_sdf_ptr(sdf_map_);
  tlplannerPtr_->set_trajopt_ptr(trajoptPtr_);
  tlplannerPtr_->set_data_ptr(dataManagerPtr_);
  tlplannerPtr_->set_rc_sdf_ptr(rc_sdf_ptr_);

  #if defined(RRT_STAR) || defined(B_RRT_STAR)
  std::cout << "before init rrt_star_ptr_" << std::endl;
  tlplannerPtr_->set_rrt_star_ptr(paraPtr_,gridmapPtr_);
  #endif

  envPtr_->set_vis_ptr(visPtr_);
  trajoptPtr_->set_vis_ptr(visPtr_);
  tlplannerPtr_->set_vis_ptr(visPtr_);

  std::vector<Eigen::Vector3d> path;
  visPtr_->visualize_path(path, "astar");
  visPtr_->visualize_pointcloud(path, "astar_vp");
  visPtr_->visualize_path(path, "car_predict");
  visPtr_->visualize_pointcloud(path, "mid_waypts");
  visPtr_->visualize_path(path, "rrt_star_final_path");
  visPtr_->visualize_pointcloud(path, "rrt_star_final_wpts");
  visPtr_->visualize_pointcloud(path, "AABB_pts");



  vector<vector<Eigen::Vector3d>> routes;
  visPtr_->visualize_path_list(routes, "rrt_star_paths");

  std::vector<std::pair<Eigen::Vector3d, double>> pcl_i;
  visPtr_->visualize_pointcloud_itensity(pcl_i, "IG_gains");

  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> lines;
  visPtr_->visualize_arrows(lines, "debug_grad_collision_p");
  visPtr_->visualize_arrows(lines, "debug_grad_collision_q");



  INFO_MSG_GREEN("[CH_PLAN] vis_ptr init done");

  // getIGPrarm();

  mani_goal_sub_ = nh_.subscribe("/mani_goal", 1, &Planner::mani_goal_cb, this);

  // airgrasp
  theta_cur_sub_ = nh_.subscribe("joint_state_est", 1, &Planner::theta_cur_cb, this);

  INFO_MSG_GREEN("[plan node] Planner Init Done.");

}

void Planner::sample_random_odom(Odom& odom_data){
  std::uniform_real_distribution<double> dist_x(odom_rand_pos_min_.x(), odom_rand_pos_max_.x());
  std::uniform_real_distribution<double> dist_y(odom_rand_pos_min_.y(), odom_rand_pos_max_.y());
  std::uniform_real_distribution<double> dist_z(odom_rand_pos_min_.z(), odom_rand_pos_max_.z());
  std::uniform_real_distribution<double> dist_yaw(odom_rand_yaw_min_, odom_rand_yaw_max_);
  std::uniform_real_distribution<double> dist_theta0(odom_rand_theta_min_.x(), odom_rand_theta_max_.x());
  std::uniform_real_distribution<double> dist_theta1(odom_rand_theta_min_.y(), odom_rand_theta_max_.y());
  std::uniform_real_distribution<double> dist_theta2(odom_rand_theta_min_.z(), odom_rand_theta_max_.z());

  odom_data.odom_p_ = Eigen::Vector3d(dist_x(rng_), dist_y(rng_), dist_z(rng_));
  double yaw = dist_yaw(rng_);
  odom_data.odom_q_ = rot_util::yaw2quaternion(yaw);
  odom_data.odom_dyaw_ = 0.0;
  odom_data.odom_v_.setZero();
  odom_data.odom_a_.setZero();
  odom_data.odom_j_.setZero();
  odom_data.odom_time_stamp_ms_ = TimeNow();

  odom_data.theta_ = Eigen::Vector3d(dist_theta0(rng_), dist_theta1(rng_), dist_theta2(rng_));
  odom_data.dtheta_.setZero();
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
    sample_random_odom(odom_data);
    theta_cur_ = odom_data.theta_;
    dtheta_cur_ = odom_data.dtheta_;
    has_theta_cur_ = true;

    Eigen::Vector3d odom_p = odom_data.odom_p_;
    Eigen::Vector3d odom_v = odom_data.odom_v_;

    Eigen::Vector3d end_pos_cur;
    Eigen::Quaterniond end_q_cur;
    rc_sdf_ptr_->getCurEndPose(odom_p, odom_data.odom_q_ ,end_pos_cur, end_q_cur);
    if (mode == HOVER){
      dataManagerPtr_->save_hover_p(odom_p, TimeNow(), dataManagerPtr_->traj_info_);
      set_state(NOTNEEDPLAN);
      INFO_MSG("[plan node] hovering...");
    }

    //! 2. 获得 地图信息
    // lock until this plan loop finish
    
    //! 3. 获得 目标状态
    Odom goal_data;
    dataManagerPtr_->get_odom(dataManagerPtr_->goal_info_, goal_data); 
    last_goal_data_ = goal_data;

    if (mode == HOVER){
      BAG(dataManagerPtr_->BagPtr_->write_float_flatmsg("/state", HOVER));
      std::this_thread::sleep_for(std::chrono::milliseconds(thread_dur));
      continue;
    }

    //! 4. 从上次执行轨迹获得初始状态
    Odom init_state;
    TrajData traj_now;

    init_state = odom_data;
    init_state.odom_a_.setZero();
    init_state.odom_dyaw_ = 0.0;
    init_state.theta_ = theta_cur_;
    std::cout << "init_state.theta_:\n" << init_state.theta_ << std::endl;
    init_state.dtheta_ = dtheta_cur_;
    std::cout << "init_p:" << init_state.odom_p_.transpose() << std::endl;

    //! 5. planner
    tlplanner::TLPlanner::PlanResState plan_res;
    bool no_need_plan = false;
    Eigen::Vector3d goal = goal_data.odom_p_;
    goal_data.odom_p_ = goal;
    goal_data.odom_v_ = Eigen::Vector3d(0.0,0.0,0.0);

    INFO_MSG("[plan node] get new goal: " << goal_data.odom_p_.transpose());
    plan_res = tlplannerPtr_->plan_goal(init_state, goal_data, traj_now);
    if (plan_res == tlplanner::TLPlanner::PlanResState::PLANSUCC){
      has_last_goal_ = true;
      last_goal_stamp_ = goal_data.odom_time_stamp_ms_;
      last_goal_ = goal_data.odom_p_;
    }
    INFO_MSG_YELLOW("traj_cur duration: " << traj_now.getTotalDuration());

    //! 6. check
    bool valid = false;
    if (plan_res == tlplanner::TLPlanner::PLANSUCC){
      valid = tlplannerPtr_->valid_cheack(traj_now);
    }
    if (valid){
      traj_now.start_time_ = TimeNow();
      dataManagerPtr_->write_traj(traj_now, dataManagerPtr_->traj_info_);
      traj_last_ = traj_now;
      has_traj_last_ = true;
      INFO_MSG_GREEN("[plan node] REPLAN SUCCESS.");
      emergency_stop_ = false;
      set_state(PLANSUCC);
    }else{
      INFO_MSG_RED("[plan node] REPLAN FAIL, EXECUTE LAST TRAJ!");
      set_state(PLANFAIL);
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
