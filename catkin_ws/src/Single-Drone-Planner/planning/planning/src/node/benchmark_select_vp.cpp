// input: map
// output: selected viewpoints


//! not used
//! not used
//! not used
//! not used

#include <ros/ros.h>
#include "plan_env_lod/sdf_map.h"
#include "nav_msgs/Odometry.h"

std::shared_ptr<fast_planner::SDFMap> sdf_map_;
ros::ServiceClient benchmark_vp_client_;

int main(int argc, char **argv)
{
    ros::init(argc, argv, "benchmark_select_vp");
    ros::NodeHandle nh;

    sdf_map_ = std::make_shared<fast_planner::SDFMap>();
    sdf_map_->initMap(nh);

    ros::Rate rate(10);

    // 两种方法计时
    double gs_time = 0.0;
    Eigen::Vector3d grid_time; grid_time.setZero();
    // grid方式步长
    Eigen::Vector3d grid_step; grid_step << 0.05, 0.1, 0.2;

    size_t cnt = 0, max_cnt = 10000;
    while(ros::ok() and cnt < max_cnt)
    {
        //! select vp
        Eigen::Vector3d cam_p;
        double yaw;
        sdf_map_->selectBenchmarkVP(cam_p, yaw);

        //! 1 GS
        double time_dur;
        // completeness_gain
        sdf_map_->getCG_GS(cam_p, yaw, time_dur);
        gs_time += time_dur;

        //! 2 grid
        for(int i = 0; i < grid_step.size(); i++)
        {
            sdf_map_->getCG_grid(cam_p, yaw, grid_step[i], time_dur);
            grid_time[i] += time_dur;
        }

        if(cnt % 100 == 0)
        {
            ROS_INFO("cnt: %d, max_cnt: %d", cnt, max_cnt)
        }

        rate.sleep();
        cnt++;
    }

    gs_time /= max_cnt;
    grid_time /= max_cnt;

    ROS_INFO("GS time: %f", gs_time);
    ROS_INFO("Grid step: %f %f %f", grid_step[0], grid_step[1], grid_step[2]);
    ROS_INFO("Grid time: %f %f %f", grid_time[0], grid_time[1], grid_time[2]);

    return 0;
}