## ReadMe
Here are 4 threads corresponding to 4 cpp files:
- tl_fsm_node.cpp
- plan_node.cpp
- traj_server_node.cpp
- data_callback.cpp

All the 4 threads are created and managed in main_node.cpp.

### plan_node.cpp
This thread is the core thread. This thread has totally split from ROS.
This thread has no-shared son class:
- <mapping::OccGridMap>
- <env::Env>
- <prediction::Predict>
- <traj_opt::TrajOpt>
- <tlplanner::TLPlanner>
The input messeges are input through class: <ShareDataManager>.
The output messeges are also saved in <ShareDataManager>.traj_info_.

The state of this thread:
  enum PlanMode{
    IDLE,
    GOAL,
    TRACK,
    LAND,
  };
  enum PlannerState{
    PLANFAIL,
    PLANSUCC,
  };

### data_callback.cpp
This thread is the data input interface. It has ros & ssdbus 2 versoin.
When turnning to ssdbus vesion, just replace the *ssdbus.cpp file in secrecy system.
The data updated by datacallback are listed in `data_callback.cpp`, as follows:
- odom_info_: Drone's odometry
- car_info_: Target car's odometry
- goal_info_: Goal's odometry
- plan_trigger_received_: plan trigger
- land_trigger_received_: land trigger
- map_msg_: map

### traj_server_node.cpp
This thread is command output interface. It has ros & ssdbus 2 versoin.
When turnning to ssdbus vesion, just replace the *ssdbus.cpp file in secrecy system.
This thread will read planned result saved in <ShareDataManager>.traj_info_, and publish control command in ros or ssdbus context.

### tl_fsm_node.cpp
This thread manages the state of plan node according to the data updated by datacallback. It can do more monitor missions in the future.
