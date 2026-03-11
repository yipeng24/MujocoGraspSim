#include "node/tl_fsm_node.h"
using rot_util = rotation_util::RotUtil;

static const char* fsm_state_names[] = {"IDLE", "HOVER", "GOAL", "TRACK", "LAND", "STOP"};

TLFSM::TLFSM(ros::NodeHandle& nh){

    std::string config_path;
    if (!nh.getParam("config_path", config_path)) {
        ROS_FATAL("[TLFSM] ~config_path ROS param required! "
                  "Add: <param name=\"config_path\" value=\"$(find planning)/config/config.yaml\"/>");
        throw std::runtime_error("[TLFSM] Missing ~config_path param");
    }
    ROS_INFO_STREAM("[TLFSM] Loading config from: " << config_path);
    para_ptr_ = std::make_shared<parameter_server::ParaeterSerer>(config_path.c_str());
    dataManagerPtr_ = std::make_shared<ShareDataManager>(para_ptr_);
    data_callbacks_ = std::make_shared<DataCallBacks>(dataManagerPtr_, para_ptr_);
    traj_server_ = std::make_shared<TrajServer>(dataManagerPtr_, para_ptr_);

    para_ptr_->get_para("fsm_mode", fsm_mode_);
    para_ptr_->get_para("land_p_x", land_dp_.x());
    para_ptr_->get_para("land_p_y", land_dp_.y());
    para_ptr_->get_para("land_p_z", land_dp_.z());

    data_callbacks_->init_ros(nh);
    traj_server_->init_ros(nh);

    vis_ptr_ = std::make_shared<vis_interface::VisInterface>(nh);

    planner_ = std::make_shared<Planner>(dataManagerPtr_, nh ,para_ptr_, vis_ptr_);

    // 发布 FSM 状态供外部显示（latched，新订阅者立即收到最新值）
    fsm_state_pub_ = nh.advertise<std_msgs::String>("fsm_state", 1, true);
}

void TLFSM::run(){
    int pub_cnt = 0;
    while (!dataManagerPtr_->s_exit_)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        switch (state_)
        {
        case IDLE:
            if (dataManagerPtr_->auto_mode_){
                INFO_MSG_YELLOW("[FSM] IDLE -> HOVER");
                planner_->set_mode(Planner::HOVER);
                state_ = HOVER;
            }
            break;
        case HOVER:
        {
            Odom odom_data;
            if (!dataManagerPtr_->get_odom(dataManagerPtr_->odom_info_, odom_data)){
                break;
            }
            if (dataManagerPtr_->auto_mode_ && fsm_mode_ == 1 && dataManagerPtr_->plan_trigger_received_){
                INFO_MSG_YELLOW("[FSM] HOVER -> GOAL");
                planner_->set_mode(Planner::GOAL);
                state_ = GOAL;
            }
            break;
        }
        case GOAL:
        {
            Odom odom_data;
            if (!dataManagerPtr_->get_odom(dataManagerPtr_->odom_info_, odom_data)){
                break;
            }
            break;
        }

        default:
            break;
        }

        // 每 10 次循环（约 10Hz）发布一次状态
        if (++pub_cnt >= 10) {
            pub_cnt = 0;
            std_msgs::String msg;
            msg.data = fsm_state_names[static_cast<int>(state_)];
            fsm_state_pub_.publish(msg);
        }
    }
    INFO_MSG_RED("[FSM] Thread Exit.");
}

bool TLFSM::set_thread_para(std::shared_ptr<std::thread>& thread, const int priority, const char* name){
    pthread_setname_np(thread->native_handle(), name);
    struct sched_param thread_param = {};
    thread_param.sched_priority = priority;
    bool succ = false;
    if(pthread_setschedparam(thread->native_handle(), SCHED_RR, &thread_param) == 0){
        INFO_MSG("Set thread priority "<<priority);
        succ = true;
    }else{
        INFO_MSG("Fail to set thread priority "<<priority);
        succ = false;
    }
    return succ;
}
