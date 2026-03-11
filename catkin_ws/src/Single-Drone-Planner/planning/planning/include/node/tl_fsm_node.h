#ifndef TL_FSM
#define TL_FSM

#include "util_gym/util_gym.hpp"
#include "util_gym/data_manager.hpp"
#include "rotation_util/rotation_util.hpp"

#include <thread>
#include <ros/ros.h>
#include <std_msgs/String.h>

#include "node/plan_node.h"
#include "node/traj_server_node.h"
#include "node/data_callback_node.h"

class TLFSM{
 public:
  enum FsmState{
    IDLE,
    HOVER,
    GOAL,
    TRACK,
    LAND,
    STOP,
  };

 private:
  int fsm_mode_;
  Eigen::Vector3d land_dp_;
  ros::Publisher fsm_state_pub_;

 public:
  std::shared_ptr<ShareDataManager> dataManagerPtr_;
  std::shared_ptr<parameter_server::ParaeterSerer> para_ptr_;
  std::shared_ptr<vis_interface::VisInterface> vis_ptr_;
  std::shared_ptr<DataCallBacks> data_callbacks_;
  std::shared_ptr<Planner> planner_;
  std::shared_ptr<TrajServer> traj_server_;

  std::shared_ptr<clutter_hand::CH_RC_SDF> rc_sdf_ptr_;

  FsmState state_ = IDLE;
 public:
  bool set_thread_para(std::shared_ptr<std::thread>& thread, const int priority, const char* name);

 public:
  TLFSM(ros::NodeHandle& nh);
  void run();

};


#endif