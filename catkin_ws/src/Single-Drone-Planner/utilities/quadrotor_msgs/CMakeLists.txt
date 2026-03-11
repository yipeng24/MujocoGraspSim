cmake_minimum_required(VERSION 3.0.2)
project(quadrotor_msgs)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  nav_msgs
  geometry_msgs
  sensor_msgs
  message_generation
)

add_message_files(
  DIRECTORY msg
  FILES
  OccMap3d.msg
  OccMap3dOld.msg
  PolyTraj.msg
  ReplanState.msg
  PositionCommand.msg
  SO3Command.msg
  AuxCommand.msg
  Px4ctrlDebug.msg
  TakeoffLand.msg
  frame.msg
  ArmAnglesState.msg
  ArmAngleState.msg
  ManiGoal.msg
  Trajectory.msg
  TrajectoryPoint.msg
  PIDParams.msg
  UAMFullState.msg
  SyncFrame.msg
  DebugOpt.msg
  PolynomialTrajectory.msg
  SimDebug.msg
  TrajEndWP.msg
  TrajSeqEndWP.msg

  # --- added for MuJoCo bridge ESC telemetry ---
  ESCTelemetryItem.msg
  ESCTelemetry.msg
)

add_service_files(
  DIRECTORY srv
  FILES
  GetBasePose.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS
    message_runtime
    std_msgs
    geometry_msgs
    nav_msgs
    sensor_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)