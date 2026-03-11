#include "quadrotor_planner_interface/quadrotor_planner_interface.h"

namespace quadrotor_planner_interface{
class TestPlanner: public QuadrotorPlannerInterface {
 public:
  using QuadrotorPlannerInterface::QuadrotorPlannerInterface;
};

} // quadrotor_planner_interface

int main(int argc, char **argv) {
  ros::init(argc, argv, "test_node");
  ros::NodeHandle nh("~");
  using namespace quadrotor_planner_interface;
  TestPlanner testPlanner(nh);
  ros::MultiThreadedSpinner spinner(4);

  double init_x, init_y, init_z;
  nh.param("planning/init_state_x", init_x, 0.0);
  nh.param("planning/init_state_y", init_y, 0.0);
  nh.param("planning/init_state_z", init_z, 0.0);
  Eigen::Vector3d init_pos(init_x, init_y, init_z);
  Eigen::Vector3d init_vel(0, 0, 0);
  Eigen::Vector3d init_acc(0, 0, 0);
  ros::Rate loopRate(10);
  ros::Time startT = ros::Time::now();
  while(ros::ok()) {
    ros::Time nowT = ros::Time::now();
    testPlanner.pubCmd(init_pos, init_vel, init_acc);
    if ((nowT-startT).toSec() > 3){
      break;
    }
    loopRate.sleep();
  }

  spinner.spin();
  return 0;
}

