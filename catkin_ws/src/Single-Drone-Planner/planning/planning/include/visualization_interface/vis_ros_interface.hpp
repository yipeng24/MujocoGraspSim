#pragma once
#include "util_gym/util_gym.hpp"
#ifdef ROS
#include <visualization/visualization.hpp>
namespace vis_interface
{
class VisInterface: public visualization::Visualization{
 public:
  VisInterface(ros::NodeHandle& nh):
               visualization::Visualization(nh){}
};


} // namespace vis_interface
#endif