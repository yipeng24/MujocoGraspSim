#include <csignal>
#include "traj_opt/traj_opt.h"
#include "util_gym/util_gym.hpp"
#include "prediction/prediction_car.hpp"
using rot_util = rotation_util::RotUtil;

std::shared_ptr<traj_opt::TrajOpt> trajoptPtr_;

bool s_exit_ = false;

void traj_opt_fnc(){
    while(!s_exit_){
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        Eigen::MatrixXd init_state, init_yaw;
        init_state.setZero(3, 4);
        init_yaw.setZero(1, 2);

        prediction::State tar;
        double t_replan = -1;

        init_state.setZero();
        init_state.col(0) << 0.495, -0.103, 1.5;
        init_state.col(1) << 0.002, -0.023, 0.004;
        init_state.col(2) << 0.0, 0.0, 0.00;
        init_state.col(3) << 0.0, 0.0, 0.0;

        init_yaw(0,0) = -0.09;
        init_yaw(0,1) = 0.0;

        tar.p_ << 1.919, -0.224, 0.252;
        tar.v_ = -0.039;
        tar.theta_ = -0.067;
        Eigen::Vector3d rpy(0, 0, -0.067);
        Eigen::Quaterniond land_q = rot_util::euler2quaternion(rpy);

        Trajectory<7> traj_land;
        INFO_MSG("init p: " << init_state.col(0).transpose());
        INFO_MSG("init v: " << init_state.col(1).transpose());
        INFO_MSG("init a: " << init_state.col(2).transpose());
        INFO_MSG("init j: " << init_state.col(3).transpose());
        INFO_MSG("init yaw: " << init_yaw);
        INFO_MSG("tar p: " << tar.p_.transpose());
        INFO_MSG("tar v: " << tar.v_);
        // INFO_MSG("tar theta: " << tar.theta_);
        // INFO_MSG("tar yaw: " << rot_util::quaternion2yaw(target_data.odom_q_));
        // INFO_MSG("tar v3d: " << Eigen::Vector3d(tar.v_*cos(tar.theta_), tar.v_*sin(tar.theta_), 0.0).transpose());
        INFO_MSG("t_replan: " << t_replan);

        // Eigen::Vector3d rpy(land_roll_, land_pitch_, rot_util::quaternion2yaw(target_data.odom_q_));
        // Eigen::Quaterniond land_q = rot_util::euler2quaternion(rpy);

        // trajoptPtr_->set_with_perception(false);
        // generate_new_traj_success = trajoptPtr_->generate_traj(init_state, init_yaw, 
        //                                                        tar.p_, Eigen::Vector3d(tar.v_*cos(tar.theta_), tar.v_*sin(tar.theta_), 0.0),
        //                                                        land_q, 8, traj_land, t_replan);

        trajoptPtr_->set_with_perception(false);
        Eigen::Vector3d v_3d = Eigen::Vector3d(tar.v_*cos(tar.theta_), tar.v_*sin(tar.theta_), 0.0);
        bool generate_new_traj_success = trajoptPtr_->generate_traj(init_state, init_yaw, 
                                                                    tar.p_, v_3d,
                                                                    land_q, 8, traj_land, -1);
    }

}

int main(int argc, char** argv){
    #ifdef ROS
    std::shared_ptr<parameter_server::ParaeterSerer> para_ptr_ = std::make_shared<parameter_server::ParaeterSerer>(
        "/home/ningshan/VM-UAV/Elastic-Tracker/src/planning/planning/config/config.yaml");
    #endif
    #ifdef SS_DBUS
    para_ptr_ = std::make_shared<parameter_server::ParaeterSerer>(
        "/blackbox/config/config.yaml");
    #endif

    trajoptPtr_ = std::make_shared<traj_opt::TrajOpt>(para_ptr_);


    std::shared_ptr<mapping::OccGridMap> rawmapPtr_ = std::make_shared<mapping::OccGridMap>();
    std::shared_ptr<map_interface::MapInterface> gridmapPtr_ = std::make_shared<map_interface::MapInterface>(rawmapPtr_);

    trajoptPtr_->set_gridmap_ptr(gridmapPtr_);


    signal(SIGINT, [](int /*sig*/){
        s_exit_ = true;
        #ifdef ROS
        ros::shutdown();
        #endif
    });

    std::shared_ptr<std::thread> thread_traj_;
    thread_traj_ = std::make_shared<std::thread>(traj_opt_fnc);

    thread_traj_->join();
    std::cout << "clean up done" << std::endl;
    std::quick_exit(0);
    return 0;
}