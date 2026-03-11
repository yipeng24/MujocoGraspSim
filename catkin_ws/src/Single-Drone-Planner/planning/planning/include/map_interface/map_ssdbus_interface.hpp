#pragma once
#include "util_gym/util_gym.hpp"
#ifdef SS_DBUS
// #include <mapping/mapping.h>
namespace map_interface
{
  
class MapInterface{
 private:
//   std::shared_ptr<mapping::OccGridMap> gridmapPtr_;
 public:
  MapInterface(){}
//   MapInterface(std::shared_ptr<mapping::OccGridMap> gridmapPtr):gridmapPtr_(gridmapPtr){}

//   inline void from_msg(const quadrotor_msgs::OccMap3d& msg) {
//     gridmapPtr_->from_msg(msg);
//   }
  inline const bool isOccupied(const Eigen::Vector3d& p) const {
    // return gridmapPtr_->isOccupied(p);
    return false;
  }
  inline const bool isOccupied(const Eigen::Vector3i& id) const {
    // return gridmapPtr_->isOccupied(id);
    return false;
  }
  inline bool checkRayValid(const Eigen::Vector3d& p0, const Eigen::Vector3d& p1) const {
    // return gridmapPtr_->checkRayValid(p0, p1);
    return true;
  }
  inline double getCostWithGrad(const Eigen::Vector3d& pos, Eigen::Vector3d& grad) const{
    // return gridmapPtr_->getCostWithGrad(pos, grad);
    grad.setZero();
    return 20.0;
  }
  inline const Eigen::Vector3i pos2idx(const Eigen::Vector3d& pt) const {
    return (pt / resolution()).array().floor().cast<int>();
  }
  inline const Eigen::Vector3d idx2pos(const Eigen::Vector3i& id) const {
    return (id.cast<double>() + Eigen::Vector3d(0.5, 0.5, 0.5)) * resolution();
  }
  inline double resolution() const{
    return 0.2;
  }
};

} // namespace map_interface
#endif