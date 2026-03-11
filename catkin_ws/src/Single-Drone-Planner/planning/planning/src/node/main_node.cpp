#include "node/tl_fsm_node.h"
#include <csignal>

bool s_exit = false;
std::shared_ptr<TLFSM> tl_fsm_;

void plan_func(std::shared_ptr<Planner> planner) {
    planner->plan_thread();
}

void traj_server_func(std::shared_ptr<TrajServer> traj_server){
    traj_server->cmd_thread();
}

void fsm_func(std::shared_ptr<TLFSM> fsm){
    fsm->run();
}


int main(int argc, char** argv){
    ros::init(argc, argv, "planning");
    ros::NodeHandle nh("~");
    tl_fsm_ = std::make_shared<TLFSM> (nh);

    signal(SIGINT, [](int /*sig*/){
        tl_fsm_->dataManagerPtr_->s_exit_ = true;
        ros::shutdown();
    });

    std::shared_ptr<std::thread> thread_plan_;
    std::shared_ptr<std::thread> thread_traj_server_;
    std::shared_ptr<std::thread> thread_fsm_;

    thread_plan_ = std::make_shared<std::thread>(plan_func, tl_fsm_->planner_);
    thread_traj_server_ = std::make_shared<std::thread>(traj_server_func, tl_fsm_->traj_server_);
    thread_fsm_ = std::make_shared<std::thread>(fsm_func, tl_fsm_);
    
    std::cout << "init thread" << std::endl;

    tl_fsm_->set_thread_para(thread_plan_, 1, "plan");
    tl_fsm_->set_thread_para(thread_traj_server_, 2, "traj_server");
    tl_fsm_->set_thread_para(thread_fsm_, 2, "fsm");

    ros::spin();

    INFO_MSG_RED("[Main node] Exit!");    

    thread_plan_->join();
    thread_traj_server_->join();
    thread_fsm_->join();

    std::cout << "clean up done" << std::endl;
    std::quick_exit(0);
    return 0;
}
