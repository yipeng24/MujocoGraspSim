/*
Copyright (C) 2022 Hongkai Ye (kyle_yeh@163.com), Longji Yin (ljyin6038@163.com )
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
OF SUCH DAMAGE.
*/
#ifndef BRRT_STAR_H
#define BRRT_STAR_H

// #include "occ_grid/occ_map.h"
#include "visualization/visualization.hpp"
#include "sampler.h"
#include "node.h"
#include "kdtree.h"
#include "map_interface/map_interface.h"
#include "util_gym/data_manager.hpp"

#include <ros/ros.h>
#include <utility>
#include <queue>
#include <algorithm>

#include <ch_rc_sdf/ch_rc_sdf.h>
namespace path_plan
{
  class BRRTStar
  {
  public:
    BRRTStar(){};
    BRRTStar(std::shared_ptr<parameter_server::ParaeterSerer>& para_ptr,
            std::shared_ptr<map_interface::MapInterface>& mapPtr) 
            : paraPtr_(para_ptr), map_ptr_(mapPtr)
    {
      paraPtr_->get_para("RRT_Star/steer_length", steer_length_);
      paraPtr_->get_para("RRT_Star/search_radius", search_radius_);
      paraPtr_->get_para("RRT_Star/search_time", search_time_);
      paraPtr_->get_para("RRT_Star/max_tree_node_nums", max_tree_node_nums_);
      paraPtr_->get_para("RRT_Star/use_informed_sampling", use_informed_sampling_);
      paraPtr_->get_para("RRT_Star/rho_yaw", rho_yaw_);
      paraPtr_->get_para("RRT_Star/rho_thetas", rho_thetas_);
      // paraPtr_->get_para("RRT_Star/use_GUILD_sampling", use_GUILD_sampling_);
      ROS_WARN_STREAM("[RRT*] param: steer_length: " << steer_length_);
      ROS_WARN_STREAM("[RRT*] param: search_radius: " << search_radius_);
      ROS_WARN_STREAM("[RRT*] param: search_time: " << search_time_);
      ROS_WARN_STREAM("[RRT*] param: max_tree_node_nums: " << max_tree_node_nums_);
      ROS_WARN_STREAM("[RRT*] param: use_informed_sampling: " << use_informed_sampling_);
      // ROS_WARN_STREAM("[RRT*] param: use_GUILD_sampling: " << use_GUILD_sampling_);
      rho_yaw_inv_ = 1.0 / rho_yaw_;
      rho_thetas_inv_ = 1.0 / rho_thetas_;

      // Eigen::Vector3d sample_range;
      // paraPtr_->get_para("RRT_Star/sample_range_x", sample_range.x());
      // paraPtr_->get_para("RRT_Star/sample_range_y", sample_range.y());
      // paraPtr_->get_para("RRT_Star/sample_range_z", sample_range.z());
      // ROS_WARN_STREAM("[RRT*] param: sample_range: " << sample_range.transpose());

      // mapping_nodelet 读取地图参数，发给 gridmapPtr_(MapInterface)
      // 此时 gridmapPtr_ 还没有地图参数
      // sampler_.setSamplingRange(Eigen::Vector3d(-0.5*sample_range.x(),-0.5*sample_range.y(),0.0), 
      //                           Eigen::Vector3d(sample_range.x(),sample_range.y(),sample_range.z()));

      valid_tree_node_nums_ = 0;
      nodes_pool_.resize(max_tree_node_nums_);
      for (int i = 0; i < max_tree_node_nums_; ++i)
      {
        nodes_pool_[i] = new TreeNode;
      }
    }
    ~BRRTStar(){};

    void setSampleingRange(const Eigen::VectorXd& min_bound, const Eigen::VectorXd& max_bound)
    {
      INFO_MSG_RED("before setSamplingRange");
      sampler_.setSamplingRange(min_bound, max_bound-min_bound);
std::cout << "[BRRT*]:bound_size: " << min_bound.size() << std::endl;

    }

    // pos + yaw + thetas
    bool plan(const Eigen::VectorXd &s, const Eigen::VectorXd &g)
    {
      reset();
      if (map_ptr_->isOccupied(s))
      {
        ROS_ERROR("[BRRT*]: Start pos collide or out of bound");
        return false;
      }
      if (map_ptr_->isOccupied(g))
      {
        ROS_ERROR("[BRRT*]: Goal pos collide or out of bound");
        return false;
      }

      if(map_ptr_->checkRayValid(s, g)){
        ROS_INFO("[BRRT*]: Start and Goal connect directly");
        final_path_.push_back(s);
        final_path_.push_back(g);

        path_list_.push_back(final_path_);
        return true;
      }

      ROS_INFO("[BRRT*]: Start and Goal not connect directly");



      std::cout << "[BRRT*]: s: " << s.transpose() << std::endl;


      Eigen::VectorXd s_cir, g_cir;
      angleForward(s, s_cir);
      angleForward(g, g_cir);

      std::cout << "[BRRT*]: s_cir: " << s_cir.transpose() << std::endl;


      /* construct start and goal nodes */
      start_node_ = nodes_pool_[1];
      start_node_->x = s_cir;
      // start_node_->yaw = s(3);
      start_node_->cost_from_start = 0.0;
      goal_node_ = nodes_pool_[0];
      goal_node_->x = g_cir;
      // goal_node_->yaw = g(3);
      goal_node_->cost_from_start = 0.0; // important
      valid_tree_node_nums_ = 2;         // put start and goal in tree

      ROS_INFO("[BRRT*]: BRRT* starts planning a path");

      /* Init the sampler for informed sampling */
      sampler_.reset();
      if(use_informed_sampling_)
      {
        calInformedSet(10000000000.0, s.head(3), g.head(3), scale_, trans_, rot_);
        sampler_.setInformedTransRot(trans_, rot_);
      }
      return brrt_star(s_cir, g_cir);
    }

    vector<Eigen::VectorXd> getPath()
    {
      return final_path_;
    }

    vector<vector<Eigen::VectorXd>> getAllPaths()
    {
      return path_list_;
    }

    vector<std::pair<double, double>> getSolutions()
    {
      return solution_cost_time_pair_list_;
    }

    void setVisualizer(const std::shared_ptr<visualization::Visualization> &visPtr)
    {
      vis_ptr_ = visPtr;

      vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> edges;
      std::vector<visualization::BALL> tree_nodes;
      vis_ptr_->visualize_balls_rrt(tree_nodes, "tree_vertice", visualization::Color::blue, 1.0);
      vis_ptr_->visualize_pairline(edges, "tree_edges", visualization::Color::red, 0.06);
    };

    void set_rc_sdf_ptr(const std::shared_ptr<clutter_hand::CH_RC_SDF> &rc_sdf_ptr)
    {
      rc_sdf_ptr_ = rc_sdf_ptr;
    }

  private:
    // nodehandle params
    // ros::NodeHandle nh_;
    std::shared_ptr<parameter_server::ParaeterSerer> paraPtr_;
    std::shared_ptr<clutter_hand::CH_RC_SDF> rc_sdf_ptr_;


    BiasSampler<Eigen::VectorXd> sampler_;
    // for informed sampling
    Eigen::Vector3d trans_, scale_;
    Eigen::Matrix3d rot_;
    bool use_informed_sampling_;

    double steer_length_;
    double search_radius_;
    double search_time_;
    int max_tree_node_nums_;
    int valid_tree_node_nums_;
    double first_path_use_time_;
    double final_path_use_time_;
    double cost_best_;

    int dim_;
    double rho_yaw_, rho_yaw_inv_;
    double rho_thetas_, rho_thetas_inv_;

    std::vector<TreeNode *> nodes_pool_;
    TreeNode *start_node_;
    TreeNode *goal_node_;
    vector<Eigen::VectorXd> final_path_;
    vector<vector<Eigen::VectorXd>> path_list_;
    vector<std::pair<double, double>> solution_cost_time_pair_list_;

    // environment
    // env::OccMap::Ptr map_ptr_;
    // std::shared_ptr<visualization::Visualization> vis_ptr_;

    std::shared_ptr<map_interface::MapInterface> map_ptr_;
    // map_interface::MapInterface::Ptr map_ptr_;
    std::shared_ptr<visualization::Visualization> vis_ptr_;


    void reset()
    {
      final_path_.clear();
      path_list_.clear();
      cost_best_ = DBL_MAX;
      solution_cost_time_pair_list_.clear();

      for (int i = 0; i < valid_tree_node_nums_; i++)
      {
        nodes_pool_[i]->parent = nullptr;
        nodes_pool_[i]->children.clear();
      }
      valid_tree_node_nums_ = 0;
    }

    void angleForward(const Eigen::VectorXd &s, Eigen::VectorXd &s_cir)
    {
      s_cir = Eigen::VectorXd(s.size()+1);
      s_cir.head(3) = s.head(3);
      s_cir(3) = rho_yaw_*cos(s(3));
      s_cir(4) = rho_yaw_*sin(s(3));
      s_cir.tail(s_cir.size()-5) = rho_thetas_*s.tail(s.size()-4);
    }

    Eigen::VectorXd angleBackward(const Eigen::VectorXd &s)
    {
      Eigen::VectorXd s_yaw(s.size()-1);
      s_yaw.head(3) = s.head(3);
      s_yaw(3) = atan2(s(4), s(3));
      s_yaw.tail(s_yaw.size()-4) = rho_thetas_inv_*s.tail(s.size()-5);
      return s_yaw;
    }

    double calDist(const Eigen::VectorXd &p1, const Eigen::VectorXd &p2)
    {
        return (p1 - p2).norm();
    }

    RRTNodeXDPtr addTreeNode(RRTNodeXDPtr &parent, const Eigen::VectorXd &state,
                             const double &cost_from_start, const double &cost_from_parent)
    {
      RRTNodeXDPtr new_node_ptr = nodes_pool_[valid_tree_node_nums_];
      valid_tree_node_nums_++;
      new_node_ptr->parent = parent;
      parent->children.push_back(new_node_ptr);
      new_node_ptr->x = state;
      new_node_ptr->cost_from_start = cost_from_start;
      new_node_ptr->cost_from_parent = cost_from_parent;
      return new_node_ptr;
    }

    void changeNodeParent(RRTNodeXDPtr &node, RRTNodeXDPtr &parent, const double &cost_from_parent)
    {
      if (node->parent)
        node->parent->children.remove(node); //DON'T FORGET THIS, remove it form its parent's children list
      node->parent = parent;
      node->cost_from_parent = cost_from_parent;
      node->cost_from_start = parent->cost_from_start + cost_from_parent;
      parent->children.push_back(node);

      // for all its descedants, change the cost_from_start and tau_from_start;
      RRTNodeXDPtr descendant(node);
      std::queue<RRTNodeXDPtr> Q;
      Q.push(descendant);
      while (!Q.empty())
      {
        descendant = Q.front();
        Q.pop();
        for (const auto &leafptr : descendant->children)
        {
          leafptr->cost_from_start = leafptr->cost_from_parent + descendant->cost_from_start;
          Q.push(leafptr);
        }
      }
    }

    void fillPath(const RRTNodeXDPtr &node_A, const RRTNodeXDPtr &node_B, vector<Eigen::VectorXd> &path)
    {
      path.clear();
      RRTNodeXDPtr node_ptr = node_A;
      while (node_ptr->parent)
      {
        path.push_back(node_ptr->x);
        node_ptr = node_ptr->parent;
      }
      path.push_back(start_node_->x);
      std::reverse(std::begin(path), std::end(path));

      node_ptr = node_B;
      while (node_ptr->parent)
      {
        path.push_back(node_ptr->x);
        node_ptr = node_ptr->parent;
      }
      path.push_back(goal_node_->x);
    }

    inline void sortNbrSet( Neighbour &nbrSet, Eigen::VectorXd &x_rand )
    {
      std::sort(nbrSet.nearing_nodes.begin(), nbrSet.nearing_nodes.end(),
              [&x_rand](NodeWithStatus &node1, NodeWithStatus &node2){
                return node1.node_ptr->cost_from_start + (node1.node_ptr->x - x_rand).norm() < 
                       node2.node_ptr->cost_from_start + (node2.node_ptr->x - x_rand).norm();
              });
    }

    inline void rewireTree( Neighbour &nbrSet, RRTNodeXDPtr &new_node, const Eigen::VectorXd &x_target)
    {
      for(auto curr_node : nbrSet.nearing_nodes)
      {
        double dist_to_potential_child = calDist(new_node->x, curr_node.node_ptr->x);
        bool not_consistent = new_node->cost_from_start + dist_to_potential_child < curr_node.node_ptr->cost_from_start ? true : false;
        bool promising = new_node->cost_from_start + dist_to_potential_child + calDist(curr_node.node_ptr->x, x_target) < cost_best_ ? true : false;
        if( not_consistent && promising )
        {
          bool connected(false);
          if (curr_node.is_checked)
            connected = curr_node.is_valid;
          else 
            connected = map_ptr_->checkRayValid(angleBackward(new_node->x), angleBackward(curr_node.node_ptr->x));
          
          if(connected)
            changeNodeParent(curr_node.node_ptr, new_node, dist_to_potential_child);
        }
      }
    }

    inline void chooseBestNode( Neighbour &nbrSet, const Eigen::VectorXd &x_rand, RRTNodeXDPtr &min_node, 
                                double &cost_start, double &cost_parent)
    {
      for( auto &curr_node : nbrSet.nearing_nodes)
      {
        // std::cout << "[BRRT*] 3.5.2" << std::endl;
        curr_node.is_checked = true;
        if(map_ptr_->checkRayValid(angleBackward(curr_node.node_ptr->x), angleBackward(x_rand)))
        {
          // std::cout << "[BRRT*] 3.5.3" << std::endl;
          curr_node.is_valid = true;
          min_node = curr_node.node_ptr;
          // std::cout << "[BRRT*] 3.5.4" << std::endl;
          cost_parent = calDist(min_node->x, x_rand);
          cost_start = min_node->cost_from_start + cost_parent;
          break;
        }else{
          curr_node.is_valid = false;
          continue;
        }
      }
    }

    bool brrt_star(const Eigen::VectorXd &s, const Eigen::VectorXd &g)
    {
      ros::Time rrt_start_time = ros::Time::now();
      bool tree_connected = false;

      // TODO
      double c_square = (g - s).squaredNorm() / 4.0;

      /* kd tree init */
      dim_ = s.rows();

// std::cout << "[BRRT*] 1.0" << std::endl;
      kdtree *treeA = kd_create(dim_);
      kdtree *treeB = kd_create(dim_);

      //Add start and goal nodes to kd trees

// std::cout << "[BRRT*] 2.0??" << std::endl;
      kd_insert(treeA, start_node_->x.data(), start_node_);
      kd_insert(treeB, goal_node_->x.data(), goal_node_);

// std::cout << "[BRRT*] 2.1" << std::endl;
      /* main loop */
      int idx = 0;
      int no_occ = 0;
      for (idx = 0; (ros::Time::now() - rrt_start_time).toSec() < search_time_ && valid_tree_node_nums_ < max_tree_node_nums_; ++idx)
      {
        // std::cout << "[BRRT*] 2.2" << std::endl;

        bool check_connect = false;
        bool selectTreeA = true;

        /* random sampling */
        Eigen::VectorXd x_rand_xd(dim_);
        sampler_.samplingOnce(x_rand_xd);
        x_rand_xd(3)*=rho_yaw_;
        x_rand_xd(4)*=rho_yaw_;
        x_rand_xd.tail(x_rand_xd.size()-5)*=rho_thetas_;

        // std::cout << "[BRRT*] 2.3" << std::endl;

        Eigen::VectorXd x_rand = x_rand_xd;

        // std::cout << "[BRRT*]: idx: " << idx << std::endl;
        if (map_ptr_->isOccupied(angleBackward(x_rand)))
        {
          // std::cout << "[BRRT*]: occ" << std::endl;
          continue;
        }
        no_occ++;
        // std::cout << "[BRRT*]: no_occ = " << no_occ << std::endl;

        /* Search neighbors in both treeA and treeB */
        Neighbour neighbour_nodesA, neighbour_nodesB;
        neighbour_nodesA.nearing_nodes.reserve(80);
        neighbour_nodesB.nearing_nodes.reserve(80);
        neighbour_nodesB.center = neighbour_nodesA.center = x_rand;

// std::cout << "[BRRT*] 3.0" << std::endl;

        // struct kdres *nbr_setA = kd_nearest_range3(treeA, x_rand[0], x_rand[1], x_rand[2], search_radius_);
        // struct kdres *nbr_setB = kd_nearest_range3(treeB, x_rand[0], x_rand[1], x_rand[2], search_radius_); 
        struct kdres *nbr_setA = kd_nearest_range(treeA, x_rand.data(), search_radius_);
        struct kdres *nbr_setB = kd_nearest_range(treeB, x_rand.data(), search_radius_); 

// std::cout << "[BRRT*] 4.0" << std::endl;

        // 没有找到邻居，就找最近点
        if ( nbr_setA == nullptr ) // TreeA
        {
          // struct kdres *p_nearest = kd_nearest3(treeA, x_rand[0], x_rand[1], x_rand[2]);
          struct kdres *p_nearest = kd_nearest(treeA, x_rand.data());
          if (p_nearest == nullptr)
          {
            ROS_ERROR("nearest query error");
            continue;
          }

// std::cout << "[BRRT*] 4.1" << std::endl;

          //! kd_tree_提取出来的需要变换回去
          RRTNodeXDPtr nearest_node = (RRTNodeXDPtr)kd_res_item_data(p_nearest);
          kd_res_free(p_nearest);

        // x_rand 得 6 维，nearest_node 得 7 维呀
          // std::cout << "tree A nearest_node: " << nearest_node->x.transpose() << std::endl;
          // std::cout << "             x_rand: " << x_rand.transpose() << std::endl;

          neighbour_nodesA.nearing_nodes.emplace_back(nearest_node, false, false);

        }else{

// std::cout << "[BRRT*] 4.2" << std::endl;

          check_connect = true;
          while (!kd_res_end(nbr_setA)){

            RRTNodeXDPtr curr_node = (RRTNodeXDPtr)kd_res_item_data(nbr_setA);

// std::cout << "tree A nearest_node: " << curr_node->x.transpose() << std::endl;
// std::cout << "             x_rand: " << x_rand.transpose() << std::endl;

            neighbour_nodesA.nearing_nodes.emplace_back(curr_node, false, false);
            // store range query result so that we dont need to query again for rewire;
            kd_res_next(nbr_setA); //go to next in kd tree range query result
          }
        }

        kd_res_free(nbr_setA); //reset kd tree range query
// std::cout << "[BRRT*] 5.0" << std::endl;

        if ( nbr_setB == nullptr )// TreeB
        {
          // struct kdres *p_nearest = kd_nearest3(treeB, x_rand[0], x_rand[1], x_rand[2]);
          struct kdres *p_nearest = kd_nearest(treeB, x_rand.data());
          if (p_nearest == nullptr)
          {
            ROS_ERROR("nearest query error");
            continue;
          }
          RRTNodeXDPtr nearest_node = (RRTNodeXDPtr)kd_res_item_data(p_nearest);
          kd_res_free(p_nearest);
          neighbour_nodesB.nearing_nodes.emplace_back(nearest_node, false, false);

          // std::cout << "tree B nearest_node: " << nearest_node->x.transpose() << std::endl;
          // std::cout << "x_rand             : " << x_rand.transpose() << std::endl;


        }else{
          check_connect = true;
          while (!kd_res_end(nbr_setB)){
            RRTNodeXDPtr curr_node = (RRTNodeXDPtr)kd_res_item_data(nbr_setB);

      // std::cout << "tree B nearest_node: " << curr_node->x.transpose() << std::endl;
      // std::cout << "x_rand             : " << x_rand.transpose() << std::endl;

            neighbour_nodesB.nearing_nodes.emplace_back(curr_node, false, false);
            // store range query result so that we dont need to query again for rewire;
            kd_res_next(nbr_setB); //go to next in kd tree range query result
          }
        }
        kd_res_free(nbr_setB); //reset kd tree range query
// std::cout << "[BRRT*] 6.0" << std::endl;

        /* Sort two neighbor sets */
        sortNbrSet(neighbour_nodesA, x_rand);
        sortNbrSet(neighbour_nodesB, x_rand);
// std::cout << "[BRRT*] 7.0" << std::endl;

        /* Get the best parent node in each tree */
        RRTNodeXDPtr min_node_A(nullptr), min_node_B(nullptr);
        double min_cost_start_A(DBL_MAX), min_cost_start_B(DBL_MAX);
        double cost_parent_A(DBL_MAX), cost_parent_B(DBL_MAX);

        chooseBestNode(neighbour_nodesA, x_rand, min_node_A, min_cost_start_A, cost_parent_A);
        chooseBestNode(neighbour_nodesB, x_rand, min_node_B, min_cost_start_B, cost_parent_B);
// std::cout << "[BRRT*] 8.0" << std::endl;
        /* Select the best tree, insert the node and rewire the tree */
        RRTNodeXDPtr new_node(nullptr);
        if( (min_node_A != nullptr) || (min_node_B != nullptr) )
        {
          if( min_cost_start_A < min_cost_start_B ){

            if(min_cost_start_A + calDist(x_rand, goal_node_->x) >= cost_best_)
              continue; // Sampling rejection

            selectTreeA = true;
            new_node = addTreeNode(min_node_A, x_rand, min_cost_start_A, cost_parent_A);
            // kd_insert3(treeA, x_rand[0], x_rand[1], x_rand[2], new_node);
            kd_insert(treeA, x_rand.data(), new_node);
            rewireTree(neighbour_nodesA, new_node, goal_node_->x);

          }else{

            if(min_cost_start_B + calDist(x_rand, start_node_->x) >= cost_best_)
              continue; // Sampling rejection
            
            selectTreeA = false;
            new_node = addTreeNode(min_node_B, x_rand, min_cost_start_B, cost_parent_B);
            // kd_insert3(treeB, x_rand[0], x_rand[1], x_rand[2], new_node);
            kd_insert(treeB, x_rand.data(), new_node);
            rewireTree(neighbour_nodesB, new_node, start_node_->x);
          }
        }
        if( (min_node_A == nullptr) || (min_node_B == nullptr) )
          check_connect = false; // No possible connection

        /* Check connection */
        if( check_connect )
        { 
          /* Accept connection if achieve better cost */
          double cost_curr =  min_cost_start_A + min_cost_start_B;
          if(cost_curr < cost_best_)
          {
            cost_best_ = cost_curr;
            tree_connected = true;
            vector<Eigen::VectorXd> curr_best_path;
            if( selectTreeA )
              fillPath(new_node, min_node_B, curr_best_path);
            else
              fillPath(min_node_A, new_node, curr_best_path);
            path_list_.emplace_back(curr_best_path);
            solution_cost_time_pair_list_.emplace_back(cost_best_, (ros::Time::now() - rrt_start_time).toSec());

            if(use_informed_sampling_)
            {
              /* Update informed set */
              scale_[0] = cost_best_ / 2.0;
              scale_[1] = sqrt(scale_[0] * scale_[0] - c_square);
              scale_[2] = scale_[1];
              sampler_.setInformedSacling(scale_);
              std::vector<visualization::ELLIPSOID> ellps;
              ellps.emplace_back(trans_, scale_, rot_);
              vis_ptr_->visualize_ellipsoids(ellps, "informed_set", visualization::yellow, 0.2);
            }
          }
        }
// std::cout << "[BRRT*] 9.0" << std::endl;

        // debug
        // visualizeWholeTree();
        // ros::Duration(0.05).sleep();

      }//End of one sampling iteration
        
      if (tree_connected)
      {
        final_path_use_time_ = (ros::Time::now() - rrt_start_time).toSec();
        ROS_INFO_STREAM("[BRRT*]: find_path_use_time: " << solution_cost_time_pair_list_.front().second << ", length: " << solution_cost_time_pair_list_.front().first);
        visualizeWholeTree();

        for(auto& path: path_list_)
          for(auto& pt: path)
            pt = angleBackward(pt);

        final_path_ = path_list_.back();
      }
      else if (valid_tree_node_nums_ == max_tree_node_nums_)
      {
        visualizeWholeTree();
        ROS_ERROR_STREAM("[BRRT*]: NOT CONNECTED TO GOAL after " << max_tree_node_nums_ << " nodes added to rrt-tree");
      }
      else
      {
        ROS_ERROR_STREAM("[BRRT*]: NOT CONNECTED TO GOAL after " << (ros::Time::now() - rrt_start_time).toSec() << " seconds");
      }
      return tree_connected;
    }

    // TODO change dim
    void visualizeWholeTree()
    {
      // std::cout << "[BRRT*]: visualizeWholeTree" << std::endl;
      // Sample and visualize the resultant tree
      vector<Eigen::VectorXd> vertice;
      vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> edges;
      vertice.clear(); edges.clear();
      sampleWholeTree(start_node_, vertice, edges);
      sampleWholeTree(goal_node_, vertice, edges);
      // std::cout << "[BRRT*]: visualizeWholeTree 1" << std::endl;
      std::vector<visualization::BALL> tree_nodes;
      tree_nodes.reserve(vertice.size());
      visualization::BALL node_p;
      node_p.radius = 0.12;
      for (size_t i = 0; i < vertice.size(); ++i)
      {
        node_p.center = vertice[i].head(3);
        vertice[i] = angleBackward(vertice[i]);
        tree_nodes.push_back(node_p);
      }
      // std::cout << "[BRRT*]: visualizeWholeTree 2" << std::endl;
      // std::cout << "vertice.size(): " << vertice.size() << std::endl;
      rc_sdf_ptr_->visRobotSeq(vertice, "tree_vertice",visualization_rc_sdf::Color::grey);
      // vis_ptr_->visualize_balls_rrt(tree_nodes, "tree_vertice", visualization::Color::blue, 1.0);
      vis_ptr_->visualize_pairline(edges, "tree_edges", visualization::Color::red, 0.06);
      // std::cout << "[BRRT*]: visualizeWholeTree 3" << std::endl;
    }

    void sampleWholeTree(const RRTNodeXDPtr &root, vector<Eigen::VectorXd> &vertice, vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> &edges)
    {
      if (root == nullptr)
        return;

      // whatever dfs or bfs
      RRTNodeXDPtr node = root;
      std::queue<RRTNodeXDPtr> Q;
      Q.push(node);
      while (!Q.empty())
      {
        node = Q.front();
        Q.pop();
        for (const auto &leafptr : node->children)
        {
          vertice.push_back(leafptr->x);
          edges.emplace_back(std::make_pair(node->x.head(3), leafptr->x.head(3)));
          Q.push(leafptr);
        }
      }
    }

    void calInformedSet(double a2, const Eigen::Vector3d &foci1, const Eigen::Vector3d &foci2,
                        Eigen::Vector3d &scale, Eigen::Vector3d &trans, Eigen::Matrix3d &rot)
    {
      trans = (foci1 + foci2) / 2.0;
      scale[0] = a2 / 2.0;
      Eigen::Vector3d diff(foci2 - foci1);
      double c_square = diff.squaredNorm() / 4.0;
      scale[1] = sqrt(scale[0] * scale[0] - c_square);
      scale[2] = scale[1];

      /* A generic implementation for SO(n) informed set */
      Eigen::Vector3d a1 = (foci2 - foci1) / calDist(foci1, foci2);
      Eigen::Matrix3d M = a1 * Eigen::MatrixXd::Identity(1,3);
      Eigen::JacobiSVD<Eigen::Matrix3d> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
      Eigen::Vector3d diag_v(1.0,svd.matrixU().determinant(),svd.matrixV().determinant());
      rot = svd.matrixU() * diag_v.asDiagonal() * svd.matrixV().transpose();
    } 


  public:
    // void samplingOnce(Eigen::Vector3d &sample)
    // {
    //   static int i = 0;
    //   sample = preserved_samples_[i];
    //   i++;
    //   i = i % preserved_samples_.size();
    // }

    // void setPreserveSamples(const vector<Eigen::Vector3d> &samples)
    // {
    //   preserved_samples_ = samples;
    // }
    // vector<Eigen::Vector3d> preserved_samples_;
  };

} // namespace path_plan
#endif