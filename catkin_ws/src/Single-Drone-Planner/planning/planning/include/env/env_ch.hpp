#pragma once
#include <Eigen/Core>
#include <memory>
#include <queue>
#include <unordered_map>
#include <rotation_util/rotation_util.hpp>
#include <parameter_server/parameter_server.hpp>
#include "map_interface/map_interface.h"

namespace std {
template <typename Scalar, int Rows, int Cols>
struct hash<Eigen::Matrix<Scalar, Rows, Cols>> {
  size_t operator()(const Eigen::Matrix<Scalar, Rows, Cols>& matrix) const {
    size_t seed = 0;
    for (size_t i = 0; i < (size_t)matrix.size(); ++i) {
      Scalar elem = *(matrix.data() + i);
      seed ^=
          std::hash<Scalar>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};
}  // namespace std

namespace env {

enum State { OPEN,
             CLOSE,
             UNVISITED };
struct Node {
  Eigen::Vector3i idx;
  bool valid = false;
  State state = UNVISITED;
  double g, h;
  Node* parent = nullptr;
};
typedef Node* NodePtr;
class NodeComparator {
 public:
  bool operator()(NodePtr& lhs, NodePtr& rhs) {
    return lhs->g + lhs->h > rhs->g + rhs->h;
  }
};


class Env {
  static constexpr int MAX_MEMORY = 1 << 18;
  static constexpr double MAX_DURATION = 0.2;

 private:
  TimePoint t_start_;

  std::unordered_map<Eigen::Vector3i, NodePtr> visited_nodes_;
  std::shared_ptr<map_interface::MapInterface> mapPtr_;
  std::shared_ptr<vis_interface::VisInterface> visPtr_;
  Eigen::Vector3d odom_p_, odom_v_;
  NodePtr data_[MAX_MEMORY];
  double desired_dist_, theta_clearance_, tolerance_d_astar_;
  double track_angle_vmax_;
  double track_angle_expect_;
  double last_desire_dis_;
  bool isdebug_ = false;
  bool is_occlusion_search_ = true;

  inline NodePtr visit(const Eigen::Vector3i& idx) {
    auto iter = visited_nodes_.find(idx);
    if (iter == visited_nodes_.end()) {
      auto ptr = data_[visited_nodes_.size()];
      ptr->idx = idx;
      ptr->valid = !mapPtr_->isOccupied(idx);
      ptr->state = UNVISITED;
      visited_nodes_[idx] = ptr;
      return ptr;
    } else {
      return iter->second;
    }
  }

  inline NodePtr visit(const Eigen::Vector3i& idx, const Eigen::Vector3i& end_idx) {
    auto iter = visited_nodes_.find(idx);
    if (iter == visited_nodes_.end()) {
      auto ptr = data_[visited_nodes_.size()];
      ptr->idx = idx;
      ptr->valid = (!mapPtr_->isOccupied(idx)) && (rayValid(idx, end_idx));
      ptr->state = UNVISITED;
      visited_nodes_[idx] = ptr;
      return ptr;
    } else {
      return iter->second;
    }
  }

 public:
  Env(std::shared_ptr<parameter_server::ParaeterSerer>& paraPtr,
      std::shared_ptr<map_interface::MapInterface>& mapPtr) : mapPtr_(mapPtr) {
    paraPtr->get_para("tracking_dis_expect", desired_dist_);
    paraPtr->get_para("track_angle_vmax", track_angle_vmax_);
    paraPtr->get_para("tolerance_d_astar", tolerance_d_astar_);
    paraPtr->get_para("theta_clearance", theta_clearance_);
    for (int i = 0; i < MAX_MEMORY; ++i) {
      data_[i] = new Node;
    }
  }
  ~Env() {
    for (int i = 0; i < MAX_MEMORY; ++i) {
      delete data_[i];
    }
  }

  inline void set_vis_ptr(std::shared_ptr<vis_interface::VisInterface> visPtr){
    visPtr_ = visPtr;
  }

  inline void set_debug(){
    isdebug_ = true;
  }

  inline void set_track_angle(const double& angle){
    track_angle_expect_ = angle;
  }

  inline void set_track_dis(const double& dis){
    desired_dist_ = dis;
  }

  inline void set_odom(const Eigen::Vector3d& odom_p, const Eigen::Vector3d& odom_v){
    odom_p_ = odom_p;
    odom_v_ = odom_v;
  }


  bool rayValid(const Eigen::Vector3i& idx0, const Eigen::Vector3i& idx1) {
    Eigen::Vector3i d_idx = idx1 - idx0;
    Eigen::Vector3i step = d_idx.array().sign().cast<int>();
    Eigen::Vector3d delta_t;
    for (int i = 0; i < 3; ++i) {
      delta_t(i) = d_idx(i) == 0 ? std::numeric_limits<double>::max() : 1.0 / std::fabs(d_idx(i));
    }
    Eigen::Vector3d t_max(0.5, 0.5, 0.5);
    t_max = t_max.cwiseProduct(delta_t);
    Eigen::Vector3i rayIdx = idx0;
    while ((rayIdx - idx1).squaredNorm() > 1) {
      if (mapPtr_->isOccupied(rayIdx)) {
        return false;
      }
      // find the shortest t_max
      int s_dim = 0;
      for (int i = 1; i < 3; ++i) {
        s_dim = t_max(i) < t_max(s_dim) ? i : s_dim;
      }
      rayIdx(s_dim) += step(s_dim);
      t_max(s_dim) += delta_t(s_dim);
    }
    return true;
  };

  inline bool findVisiblePath(const Eigen::Vector3i& start_idx,
                              const Eigen::Vector3i& end_idx,
                              std::vector<Eigen::Vector3i>& idx_path,
                              bool is_use_viewpoint = false) {
    double stop_dist = desired_dist_ / mapPtr_->resolution();
    auto stopCondition = [&](const NodePtr& ptr) -> bool {
      return ptr->h < tolerance_d_astar_ / mapPtr_->resolution() && rayValid(ptr->idx, end_idx);
    };
    auto calulateHeuristic = [&](const NodePtr& ptr) {
      Eigen::Vector3i dp = end_idx - ptr->idx;
      if (!is_use_viewpoint){
        double dr = dp.head(2).norm();
        double lambda = 1 - stop_dist / dr;
        double dx = lambda * dp.x();
        double dy = lambda * dp.y();
        double dz = dp.z();
        ptr->h = fabs(dx) + fabs(dy) + abs(dz);
        double dx0 = (start_idx - end_idx).x();
        double dy0 = (start_idx - end_idx).y();
        double cross = fabs(dx * dy0 - dy * dx0) + abs(dz);
        ptr->h += 0.001 * cross;
      }else{
        Eigen::Vector3i dp = end_idx - ptr->idx;
        int dx = dp.x();
        int dy = dp.y();
        int dz = dp.z();
        ptr->h = abs(dx) + abs(dy) + abs(dz);
        double dx0 = (start_idx - end_idx).x();
        double dy0 = (start_idx - end_idx).y();
        double cross = fabs(dx * dy0 - dy * dx0) + abs(dz);
        ptr->h += 0.001 * cross;
      }
    };

    // initialization of datastructures
    std::priority_queue<NodePtr, std::vector<NodePtr>, NodeComparator> open_set;
    std::vector<std::pair<Eigen::Vector3i, double>> neighbors;
    // NOTE 6-connected graph
    for (int i = 0; i < 3; ++i) {
      Eigen::Vector3i neighbor(0, 0, 0);
      neighbor[i] = 1;
      neighbors.emplace_back(neighbor, 1);
      neighbor[i] = -1;
      neighbors.emplace_back(neighbor, 1);
    }
    bool ret = false;
    NodePtr curPtr = visit(start_idx);
    // NOTE we should permit the start pos invalid! (for corridor generation)
    if (!curPtr->valid) {
      visited_nodes_.clear();
      std::cout << "start postition invalid!" << std::endl;
      return false;
    }
    curPtr->parent = nullptr;
    curPtr->g = 0;
    calulateHeuristic(curPtr);
    curPtr->state = CLOSE;

    double t_cost = durationSecond(TimeNow(), t_start_); 
    if (t_cost > MAX_DURATION) {
      std::cout << "[env] search costs more than " << MAX_DURATION << "s!" << std::endl;
    }
    while (visited_nodes_.size() < MAX_MEMORY && t_cost <= MAX_DURATION) {
      for (const auto& neighbor : neighbors) {
        auto neighbor_idx = curPtr->idx + neighbor.first;
        auto neighbor_dist = neighbor.second;
        NodePtr neighborPtr;
        if (is_occlusion_search_){
          neighborPtr = visit(neighbor_idx, end_idx);
        }else{
          neighborPtr = visit(neighbor_idx);
        }
         
        if (neighborPtr->state == CLOSE) {
          continue;
        }
        if (neighborPtr->state == OPEN) {
          // check neighbor's g score
          // determine whether to change its parent to current
          if (neighborPtr->g > curPtr->g + neighbor_dist) {
            neighborPtr->parent = curPtr;
            neighborPtr->g = curPtr->g + neighbor_dist;
          }
          continue;
        }
        if (neighborPtr->state == UNVISITED) {
          if (neighborPtr->valid) {
            neighborPtr->parent = curPtr;
            neighborPtr->state = OPEN;
            neighborPtr->g = curPtr->g + neighbor_dist;
            calulateHeuristic(neighborPtr);
            open_set.push(neighborPtr);
          }
        }
      }  // for each neighbor
      if (open_set.empty()) {
        std::cout << "[env] no way!" << std::endl;
        break;
      }
      curPtr = open_set.top();
      open_set.pop();
      curPtr->state = CLOSE;
      if (stopCondition(curPtr)) {
        ret = true;
        break;
      }
      if (visited_nodes_.size() == MAX_MEMORY) {
        std::cout << "[env] out of memory!" << std::endl;
      }
    }
    if (ret) {
      for (NodePtr ptr = curPtr; ptr != nullptr; ptr = ptr->parent) {
        idx_path.push_back(ptr->idx);
      }
      // idx_path.push_back(start_idx);
      std::reverse(idx_path.begin(), idx_path.end());
    }
    visited_nodes_.clear();

    return ret;
  }

  Eigen::Vector3i get_viewpoint_idx(const Eigen::Vector3d& target){
    Eigen::Vector3d viewpoint = target + Eigen::Vector3d(cos(track_angle_expect_), sin(track_angle_expect_), 0.0) * desired_dist_;
    Eigen::Vector3i end_idx = mapPtr_->pos2idx(viewpoint);
    // TODO when viewpoint in occ
    if(mapPtr_->isOccupied(end_idx)){
      ;
    }
    return end_idx;
  }



  Eigen::Vector3d get_viewpoint(const Eigen::Vector3d& target, const double track_angle_expect){
    double desired_dist = last_desire_dis_ * 1.25 > desired_dist_ ? desired_dist_ : last_desire_dis_ * 1.25;
    // INFO_MSG("desired_dist_init: " << desired_dist);
    Eigen::Vector3d viewpoint = target + Eigen::Vector3d(cos(track_angle_expect_), sin(track_angle_expect_), 0.0) * desired_dist;
    // INFO_MSG("check ray: " << target.transpose()<<", vp: " << viewpoint.transpose()<<": "<<mapPtr_->checkRayValid(target, viewpoint));
    // TODO when viewpoint in occ
    while (mapPtr_->isOccupied(viewpoint) || mapPtr_->checkRayValid(target, viewpoint) == false){
      desired_dist = desired_dist > mapPtr_->resolution() ? desired_dist * 0.75 : 0.0;
      viewpoint = target + Eigen::Vector3d(cos(track_angle_expect_), sin(track_angle_expect_), 0.0) * desired_dist;
    }
    // INFO_MSG("desired_dist_fix: " << desired_dist);
    last_desire_dis_ = desired_dist;
    return viewpoint;
  }

  void getViewpointVec(const Eigen::Vector3d& start_p,
                       const std::vector<Eigen::Vector3d>& targets,
                       const double target_dt,
                       std::vector<Eigen::Vector3d>& viewpoints){
    viewpoints.clear();
    assert(targets.size() > 0);
    double track_angle_expect = track_angle_expect_;
    Eigen::Vector3d tar2start = start_p - targets[0];
    double start_angle = atan2(tar2start.y(),tar2start.x());
    last_desire_dis_ = desired_dist_;
    viewpoints.push_back(get_viewpoint(targets[0], start_angle));
    // INFO_MSG("targets_"<<0<<": "<<targets[0].transpose()<<  ", vp: " << viewpoints.back().transpose());
    double last_angle = start_angle;
    
    if(targets.size() <= 1) return;
    for(size_t i = 1; i < targets.size(); i++){
      double exp_angle = rotation_util::RotUtil::truncate_error_angle(last_angle, track_angle_expect, target_dt*track_angle_vmax_);
      Eigen::Vector3d viewpoint = get_viewpoint(targets[i], exp_angle);
      viewpoints.push_back(viewpoint);
      // INFO_MSG("targets_"<<i<<": "<<targets[i].transpose()<<  ", vp: " << viewpoints.back().transpose());
      last_angle = exp_angle;
    }
  }

  inline bool findVisibleStartIdx(const Eigen::Vector3i start_idx, 
                                  const Eigen::Vector3i last_end_idx, 
                                  const Eigen::Vector3i end_idx, 
                                  Eigen::Vector3i& new_start_idx)
  {
    Eigen::Vector3d start_p = mapPtr_->idx2pos(start_idx);
    Eigen::Vector3d end_p = mapPtr_->idx2pos(end_idx);
    Eigen::Vector3d last_end_p = mapPtr_->idx2pos(last_end_idx);
    Eigen::Vector3d search_dir = (last_end_p - start_p).normalized();
    double tar_dis = (last_end_p - start_p).norm();
    double search_dis = tar_dis;
    Eigen::Vector3d new_start_p = start_p;
    // find the visible point on line[startp -> last_endp]
    while (!mapPtr_->checkRayValid(new_start_p, end_p)){
      if ((last_end_p - new_start_p).norm() <= mapPtr_->resolution()){ // stop condition
        new_start_p = last_end_p;
        if (mapPtr_->isOccupied(new_start_p)) return false;
      }
      search_dis = std::max(search_dis / 2.0 ,mapPtr_->resolution());
      new_start_p = new_start_p + search_dir * search_dis;
    }
    // extend the visible point along line[endp -> new_startp]
    search_dir = (new_start_p - end_p).normalized();
    Eigen::Vector3d next_new_start_p = new_start_p + search_dir * mapPtr_->resolution();
    tar_dis = (next_new_start_p - end_p).norm();
    if (tar_dis > desired_dist_ - mapPtr_->resolution()){
      tar_dis = desired_dist_;
      new_start_p = end_p + search_dir * desired_dist_;
    }else{
      while (!mapPtr_->isOccupied(next_new_start_p)){
        new_start_p = next_new_start_p;
        next_new_start_p = new_start_p + search_dir * mapPtr_->resolution();
        tar_dis = (next_new_start_p - end_p).norm();
        if (tar_dis >= desired_dist_) break;
      }
    }

    INFO_MSG("findVisibleStart dis: " << tar_dis << " | " << desired_dist_);
    new_start_idx = mapPtr_->pos2idx(new_start_p);
    return true;
  }

  inline bool findVisiblePath(const Eigen::Vector3d& start_p,
                              const std::vector<Eigen::Vector3d>& targets,
                              const double target_dt,
                              std::vector<Eigen::Vector3d>& way_pts,
                              std::vector<Eigen::Vector3d>& path,
                              bool is_use_viewpoint = false) {
    t_start_ = TimeNow();
    Eigen::Vector3i start_idx = mapPtr_->pos2idx(start_p);
    std::vector<Eigen::Vector3i> idx_path;
    path.push_back(start_p);

    std::vector<Eigen::Vector3d> viewpoints;
    if (is_use_viewpoint){
      getViewpointVec(start_p, targets, target_dt, viewpoints);
      visPtr_->visualize_pointcloud(viewpoints, "astar_vp");
    }else{
      viewpoints = targets;
    }

    // try occlusion search
    bool occlusion_search_succ = true;
    is_occlusion_search_ = true;
    for (size_t i = 0; i < viewpoints.size(); i++) {
      Eigen::Vector3d target = viewpoints[i];
      Eigen::Vector3i end_idx = mapPtr_->pos2idx(target);
      idx_path.clear();
      // ensure the start_idx is visible the end_idx
      Eigen::Vector3i new_start_idx;
      if (i == 0){
        if (!rayValid(start_idx, end_idx)){
          // occlusion_search_succ = false;
          std::cout << "[env] first sart is not Visible!" << std::endl;
          // break;
          new_start_idx = start_idx;
          is_occlusion_search_ = false;
          is_use_viewpoint = true;
        }else{
          new_start_idx = start_idx;
          is_occlusion_search_ = true;
          is_use_viewpoint = false;
        }
        if (!findVisiblePath(start_idx, end_idx, idx_path, is_use_viewpoint)) {
          occlusion_search_succ = false;
          break;
        }
      }
      if (i > 0){
        std::cout << "[env] search " << i << std::endl;
        is_occlusion_search_ = false;
        is_use_viewpoint = true;
        Eigen::Vector3i last_end_idx = mapPtr_->pos2idx(viewpoints[i-1]);
        if(!findVisibleStartIdx(start_idx, last_end_idx, end_idx, new_start_idx)){
          std::cout << "[env] find Visible Start fail!" << std::endl;
          occlusion_search_succ = false;
          break;
        }
        if (!findVisiblePath(start_idx, new_start_idx, idx_path, is_use_viewpoint)) {
          occlusion_search_succ = false;
          break;
        }
      }
    

      start_idx = idx_path.back();
      for (const auto& idx : idx_path) {
        path.push_back(mapPtr_->idx2pos(idx));
      }
      way_pts.push_back(mapPtr_->idx2pos(start_idx));
    }
    
    if (!occlusion_search_succ){
      INFO_MSG_RED("[Env] occlusion_search fail");
      path.clear();
      path.push_back(start_p);
      way_pts.clear();
      is_occlusion_search_ = false;
      is_use_viewpoint = false;
      for (size_t i = 0; i < viewpoints.size(); i++) {
        Eigen::Vector3d target = viewpoints[i];
        Eigen::Vector3i end_idx = mapPtr_->pos2idx(target);
        idx_path.clear();
        if (!findVisiblePath(start_idx, end_idx, idx_path, is_use_viewpoint)) {
          return false;
        }
        start_idx = idx_path.back();
        for (const auto& idx : idx_path) {
          path.push_back(mapPtr_->idx2pos(idx));
        }
        way_pts.push_back(mapPtr_->idx2pos(start_idx));
      }
    }
    double t_cost = durationSecond(TimeNow(), t_start_); 
    INFO_MSG("[env] A star time: " << t_cost*1e3 << " ms");
    return true;
  }

  inline bool astar_search(const Eigen::Vector3i& start_idx,
                           const Eigen::Vector3i& end_idx,
                           std::vector<Eigen::Vector3i>& idx_path) {
    auto stopCondition = [&](const NodePtr& ptr) -> bool {
      return ptr->h < desired_dist_ / mapPtr_->resolution();
    };
    auto calulateHeuristic = [&](const NodePtr& ptr) {
      Eigen::Vector3i dp = end_idx - ptr->idx;
      int dx = dp.x();
      int dy = dp.y();
      int dz = dp.z();
      ptr->h = abs(dx) + abs(dy) + abs(dz);
      double dx0 = (start_idx - end_idx).x();
      double dy0 = (start_idx - end_idx).y();
      double cross = fabs(dx * dy0 - dy * dx0) + abs(dz);
      ptr->h += 0.001 * cross;
    };
    // initialization of datastructures
    std::priority_queue<NodePtr, std::vector<NodePtr>, NodeComparator> open_set;
    std::vector<std::pair<Eigen::Vector3i, double>> neighbors;
    // NOTE 6-connected graph
    for (int i = 0; i < 3; ++i) {
      Eigen::Vector3i neighbor(0, 0, 0);
      neighbor[i] = 1;
      neighbors.emplace_back(neighbor, 1);
      neighbor[i] = -1;
      neighbors.emplace_back(neighbor, 1);
    }
    bool ret = false;
    NodePtr curPtr = visit(start_idx);
    // NOTE we should permit the start pos invalid! (for corridor generation)
    if (!curPtr->valid) {
      visited_nodes_.clear();
      std::cout << "start postition invalid!" << std::endl;
      return false;
    }
    curPtr->parent = nullptr;
    curPtr->g = 0;
    calulateHeuristic(curPtr);
    curPtr->state = CLOSE;

    while (visited_nodes_.size() < MAX_MEMORY) {
      for (const auto& neighbor : neighbors) {
        auto neighbor_idx = curPtr->idx + neighbor.first;
        auto neighbor_dist = neighbor.second;
        NodePtr neighborPtr = visit(neighbor_idx);
        if (neighborPtr->state == CLOSE) {
          continue;
        }
        if (neighborPtr->state == OPEN) {
          // check neighbor's g score
          // determine whether to change its parent to current
          if (neighborPtr->g > curPtr->g + neighbor_dist) {
            neighborPtr->parent = curPtr;
            neighborPtr->g = curPtr->g + neighbor_dist;
          }
          continue;
        }
        if (neighborPtr->state == UNVISITED) {
          if (neighborPtr->valid) {
            neighborPtr->parent = curPtr;
            neighborPtr->state = OPEN;
            neighborPtr->g = curPtr->g + neighbor_dist;
            calulateHeuristic(neighborPtr);
            open_set.push(neighborPtr);
          }
        }
      }  // for each neighbor
      if (open_set.empty()) {
        std::cout << "[astar search] no way!" << std::endl;
        break;
      }
      curPtr = open_set.top();
      open_set.pop();
      curPtr->state = CLOSE;
      if (stopCondition(curPtr)) {
        ret = true;
        break;
      }
      if (visited_nodes_.size() == MAX_MEMORY) {
        std::cout << "[astar search] out of memory!" << std::endl;
      }
    }
    if (ret) {
      for (NodePtr ptr = curPtr; ptr != nullptr; ptr = ptr->parent) {
        idx_path.push_back(ptr->idx);
      }
      // idx_path.push_back(start_idx);
      std::reverse(idx_path.begin(), idx_path.end());
    }
    visited_nodes_.clear();

    return ret;
  }

  inline bool astar_search(const Eigen::Vector3d& start_p,
                           const Eigen::Vector3d& end_p,
                           std::vector<Eigen::Vector3d>& path) {
    Eigen::Vector3i start_idx = mapPtr_->pos2idx(start_p);
    Eigen::Vector3i end_idx = mapPtr_->pos2idx(end_p);
    std::vector<Eigen::Vector3i> idx_path;
    bool ret = astar_search(start_idx, end_idx, idx_path);
    path.clear();
    for (const auto& id : idx_path) {
      path.push_back(mapPtr_->idx2pos(id));
    }
    return ret;
  }

  inline void visible_pair(const Eigen::Vector3d& center,
                           Eigen::Vector3d& seed,
                           Eigen::Vector3d& visible_p,
                           double& theta) {
    Eigen::Vector3d dp = seed - center;
    double theta0 = atan2(dp.y(), dp.x());
    double d_theta = mapPtr_->resolution() / desired_dist_ / 2;
    double t_l, t_r;
    for (t_l = theta0 - d_theta; t_l > theta0 - M_PI; t_l -= d_theta) {
      Eigen::Vector3d p = center;
      p.x() += desired_dist_ * cos(t_l);
      p.y() += desired_dist_ * sin(t_l);
      if (!mapPtr_->checkRayValid(p, center)) {
        t_l += d_theta;
        break;
      }
    }
    for (t_r = theta0 + d_theta; t_r < theta0 + M_PI; t_r += d_theta) {
      Eigen::Vector3d p = center;
      p.x() += desired_dist_ * cos(t_r);
      p.y() += desired_dist_ * sin(t_r);
      if (!mapPtr_->checkRayValid(p, center)) {
        t_r -= d_theta;
        break;
      }
    }
    double theta_v = (t_l + t_r) / 2;
    visible_p = center;
    visible_p.x() += desired_dist_ * cos(theta_v);
    visible_p.y() += desired_dist_ * sin(theta_v);
    theta = (t_r - t_l) / 2;
    double theta_c = theta < theta_clearance_ ? theta : theta_clearance_;
    if (theta0 - t_l < theta_c) {
      seed = center;
      seed.x() += desired_dist_ * cos(t_l + theta_c);
      seed.y() += desired_dist_ * sin(t_l + theta_c);
    } else if (t_r - theta0 < theta_c) {
      seed = center;
      seed.x() += desired_dist_ * cos(t_r - theta_c);
      seed.y() += desired_dist_ * sin(t_r - theta_c);
    }
    return;
  }

  inline void generate_visible_regions(const std::vector<Eigen::Vector3d>& targets,
                                       std::vector<Eigen::Vector3d>& seeds,
                                       std::vector<Eigen::Vector3d>& visible_ps,
                                       std::vector<double>& thetas) {
    assert(targets.size() == seeds.size());
    visible_ps.clear();
    thetas.clear();
    Eigen::Vector3d visible_p;
    double theta = 0;
    int M = targets.size();
    for (int i = 0; i < M; ++i) {
      visible_pair(targets[i], seeds[i], visible_p, theta);
      visible_ps.push_back(visible_p);
      thetas.push_back(theta);
    }
    return;
  }

  // NOTE
  // predict -> generate a series of circles
  // from drone to circle one by one search visible
  // put them together to generate corridor
  // for each center of circle, generate a visible region
  // optimization penalty:  <a,b>-cos(theta0)*|a|*|b| <= 0

  inline bool short_astar(const Eigen::Vector3d& start_p,
                          const Eigen::Vector3d& end_p,
                          std::vector<Eigen::Vector3d>& path) {
    Eigen::Vector3i start_idx = mapPtr_->pos2idx(start_p);
    Eigen::Vector3i end_idx = mapPtr_->pos2idx(end_p);
    if (start_idx == end_idx) {
      path.clear();
      path.push_back(start_p);
      path.push_back(end_p);
      return true;
    }
    auto stopCondition = [&](const NodePtr& ptr) -> bool {
      return ptr->idx == end_idx;
    };
    auto calulateHeuristic = [&](const NodePtr& ptr) {
      Eigen::Vector3i dp = end_idx - ptr->idx;
      int dx = dp.x();
      int dy = dp.y();
      int dz = dp.z();
      ptr->h = abs(dx) + abs(dy) + abs(dz);
      double dx0 = (start_idx - end_idx).x();
      double dy0 = (start_idx - end_idx).y();
      double cross = fabs(dx * dy0 - dy * dx0) + abs(dz);
      ptr->h += 0.001 * cross;
    };
    // initialization of datastructures
    std::priority_queue<NodePtr, std::vector<NodePtr>, NodeComparator> open_set;
    std::vector<std::pair<Eigen::Vector3i, double>> neighbors;
    // NOTE 6-connected graph
    for (int i = 0; i < 3; ++i) {
      Eigen::Vector3i neighbor(0, 0, 0);
      neighbor[i] = 1;
      neighbors.emplace_back(neighbor, 1);
      neighbor[i] = -1;
      neighbors.emplace_back(neighbor, 1);
    }
    bool ret = false;
    NodePtr curPtr = visit(start_idx);
    // NOTE we should permit the start pos invalid! (for corridor generation)
    if (!curPtr->valid) {
      visited_nodes_.clear();
      std::cout << "[short astar]start postition invalid!" << std::endl;
      return false;
    }

    if (!visit(end_idx)->valid) {
      visited_nodes_.clear();
      INFO_MSG_RED("end postition invalid!");
      return false;
    }
    curPtr->parent = nullptr;
    curPtr->g = 0;
    calulateHeuristic(curPtr);
    curPtr->state = CLOSE;
    INFO_MSG("start_inx: " << start_idx.transpose());
    INFO_MSG("end_idx: " << end_idx.transpose());
    if (stopCondition(curPtr)) {
      INFO_MSG_RED("start == end!");
      path.push_back(mapPtr_->idx2pos(start_idx));
      path.push_back(mapPtr_->idx2pos(end_idx));
      return true;
    }

    while (visited_nodes_.size() < MAX_MEMORY) {
      for (const auto& neighbor : neighbors) {
        auto neighbor_idx = curPtr->idx + neighbor.first;
        auto neighbor_dist = neighbor.second;
        NodePtr neighborPtr = visit(neighbor_idx);
        if (neighborPtr->state == CLOSE) {
          continue;
        }
        if (neighborPtr->state == OPEN) {
          // check neighbor's g score
          // determine whether to change its parent to current
          if (neighborPtr->g > curPtr->g + neighbor_dist) {
            neighborPtr->parent = curPtr;
            neighborPtr->g = curPtr->g + neighbor_dist;
          }
          continue;
        }
        if (neighborPtr->state == UNVISITED) {
          if (neighborPtr->valid) {
            neighborPtr->parent = curPtr;
            neighborPtr->state = OPEN;
            neighborPtr->g = curPtr->g + neighbor_dist;
            calulateHeuristic(neighborPtr);
            open_set.push(neighborPtr);
          }
        }
      }  // for each neighbor
      if (open_set.empty()) {
        std::cout << "[short astar] no way!" << std::endl;
        break;
      }
      curPtr = open_set.top();
      open_set.pop();
      curPtr->state = CLOSE;
      // std::cout << "open set top: "<<  mapPtr_->idx2pos(curPtr->idx).transpose()  << std::endl;
      // std::cout << "open set top: "<<  curPtr->idx.transpose()  << std::endl;
      if (stopCondition(curPtr)) {
        ret = true;
        break;
      }
      if (visited_nodes_.size() == MAX_MEMORY) {
        std::cout << "[short astar] out of memory!" << std::endl;
      }
    }
    if (ret) {
      INFO_MSG("ret = "<<ret);
      for (NodePtr ptr = curPtr; ptr != nullptr; ptr = ptr->parent) {
        path.push_back(mapPtr_->idx2pos(ptr->idx));
      }
      // for (auto p: path){
      //   INFO_MSG("p: " << p.transpose());
      // }

      std::reverse(path.begin(), path.end());
    }
    visited_nodes_.clear();
    return ret;
  }

  inline void pts2path(const std::vector<Eigen::Vector3d>& wayPts, std::vector<Eigen::Vector3d>& path) {
    path.clear();
    path.push_back(wayPts.front());
    int M = wayPts.size();
    std::vector<Eigen::Vector3d> short_path;
    for (int i = 0; i < M - 1; ++i) {
      const Eigen::Vector3d& p0 = path.back();
      const Eigen::Vector3d& p1 = wayPts[i + 1];
      if (mapPtr_->pos2idx(p0) == mapPtr_->pos2idx(p1)) {
        continue;
      }
      if (!mapPtr_->checkRayValid(p0, p1)) {
        short_path.clear();
        short_astar(p0, p1, short_path);
        for (const auto& p : short_path) {
          path.push_back(p);
        }
      }
      path.push_back(p1);
    }
    if (path.size() < 2) {
      Eigen::Vector3d p = path.front();
      p.z() += 0.1;
      path.push_back(p);
    }
  }
};

}  // namespace env