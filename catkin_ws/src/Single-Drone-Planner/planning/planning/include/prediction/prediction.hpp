#pragma once

#include <mapping/mapping.h>
#include <ros/ros.h>
#include <visualization/visualization.hpp>
#include <Eigen/Core>
#include <queue>

namespace prediction {

struct Node {
  Eigen::Vector3d p, v, a;
  double t;
  double score;
  double h;
  Node* parent = nullptr;
};
typedef Node* NodePtr;
class NodeComparator {
 public:
  bool operator()(NodePtr& lhs, NodePtr& rhs) {
    return lhs->score + lhs->h > rhs->score + rhs->h;
  }
};
struct Predict {
 private:
  static constexpr int MAX_MEMORY = 1 << 22;
  // searching

  double dt;
  double rho_a;
  double car_z, vmax;
  mapping::OccGridMap map;
  std::shared_ptr<visualization::Visualization> visPtr_;
  NodePtr data[MAX_MEMORY];
  int stack_top;
  bool is_debug_ = false;

  inline bool isValid(const Eigen::Vector3d& ps, const Eigen::Vector3d& pt, const Eigen::Vector3d& v) const {
    if (v.norm() >= vmax){
      return false;
    }
    if (!map.checkRayValid(ps, pt) || map.isOccupied(pt)){
      return false;
    }
    return true;
  }

 public:
  inline Predict(ros::NodeHandle& nh) {
    nh.getParam("tracking_dt", dt);
    nh.getParam("prediction/rho_a", rho_a);
    nh.getParam("prediction/vmax", vmax);
    nh.getParam("prediction/pause_debug", is_debug_);
    for (int i = 0; i < MAX_MEMORY; ++i) {
      data[i] = new Node;
    }
  }
  inline void setMap(const mapping::OccGridMap& _map) {
    map = _map;
    // map.inflate_last();
  }
  inline void set_vis_ptr(std::shared_ptr<visualization::Visualization> visPtr){
    visPtr_ = visPtr;
  }
  inline bool predict(const Eigen::Vector3d& target_p,
                      const Eigen::Vector3d& target_v,
                      std::vector<Eigen::Vector3d>& target_predcit, const double& pre_dur,
                      const double& max_time = 0.1) {
    auto score = [&](const NodePtr& ptr) -> double {
      return rho_a * ptr->a.norm();
    };
    Eigen::Vector3d end_p = target_p + target_v * pre_dur;
    auto calH = [&](const NodePtr& ptr) -> double {
      return 0.001 * (ptr->p - end_p).norm();
    };
    ros::Time t_start = ros::Time::now();
    std::priority_queue<NodePtr, std::vector<NodePtr>, NodeComparator> open_set;

    Eigen::Vector3d input(0, 0, 0);

    stack_top = 0;
    NodePtr curPtr = data[stack_top++];
    curPtr->p = target_p;
    curPtr->v = target_v;
    curPtr->a.setZero();
    curPtr->parent = nullptr;
    curPtr->score = 0;
    curPtr->h = 0;
    curPtr->t = 0;
    double dt2_2 = dt * dt / 2;
    while (curPtr->t < pre_dur) {
      for (input.x() = -3; input.x() <= 3; input.x() += 3)
        for (input.y() = -3; input.y() <= 3; input.y() += 3) {
          Eigen::Vector3d p = curPtr->p + curPtr->v * dt + input * dt2_2;
          Eigen::Vector3d v = curPtr->v + input * dt;
          if (is_debug_){
            INFO_MSG("curPtr: "<<curPtr->p.transpose()<<", input: "<<input.transpose()<<", p: "<<p.transpose());
            vis_openset(open_set);
            vis_curnode(curPtr);
            int a;
            std::cin >> a;
          }
          if (!isValid(curPtr->p, p, v)) {
            INFO_MSG("NO Valid");
            continue;
          }
          if (stack_top == MAX_MEMORY) {
            std::cout << "[prediction] out of memory!" << std::endl;
            return false;
          }
          double t_cost = (ros::Time::now() - t_start).toSec();
          if (!is_debug_ && t_cost > max_time) {
            std::cout << "[prediction] too slow!" << std::endl;
            return false;
          }
          NodePtr ptr = data[stack_top++];
          ptr->p = p;
          ptr->v = v;
          ptr->a = input;
          ptr->parent = curPtr;
          ptr->t = curPtr->t + dt;
          ptr->score = curPtr->score + score(ptr);
          ptr->h = calH(ptr);
          open_set.push(ptr);
          // std::cout << "open set push: " << p.transpose() << std::endl;
        }
      if (open_set.empty()) {
        std::cout << "[prediction] no way!" << std::endl;
        return false;
      }
      curPtr = open_set.top();
      open_set.pop();
    }
    target_predcit.clear();
    while (curPtr != nullptr) {
      target_predcit.push_back(curPtr->p);
      curPtr = curPtr->parent;
    }
    std::reverse(target_predcit.begin(), target_predcit.end());
    return true;
  }

  void vis_openset(std::priority_queue<NodePtr, std::vector<NodePtr>, NodeComparator> open_set){
    std::vector<Eigen::Vector3d> open_pts;
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> connects;
    while (!open_set.empty()){
      NodePtr curPtr = open_set.top();
      open_set.pop();
      open_pts.push_back(curPtr->p);
      if (curPtr->parent != nullptr) connects.emplace_back(curPtr->parent->p, curPtr->p);
    }
    visPtr_->visualize_pointcloud(open_pts, "openset");
    visPtr_->visualize_pairline(connects, "openset_con");
  }

  void vis_curnode(NodePtr curnode){
    std::vector<Eigen::Vector3d> pts;
    pts.push_back(curnode->p);
    visPtr_->visualize_pointcloud(pts, "curnode");
  }
};

}  // namespace prediction
