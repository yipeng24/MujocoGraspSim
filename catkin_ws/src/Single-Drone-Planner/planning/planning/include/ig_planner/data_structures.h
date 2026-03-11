#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H

#include <ros/ros.h>
#include <Eigen/Eigen>

class RRTNode
{
public:
  Eigen::Vector4d state_;
  RRTNode* parent_;
  std::vector<RRTNode*> children_;
  double gain_;
  bool gain_explicitly_calculated_;

  RRTNode() : parent_(NULL), gain_(0.0), gain_explicitly_calculated_(false)
  {
  }
  ~RRTNode()
  {
    for (typename std::vector<RRTNode*>::iterator node_it = children_.begin();
         node_it != children_.end(); ++node_it)
    {
      delete (*node_it);
      (*node_it) = NULL;
    }
  }

  RRTNode* getCopyOfParentBranch()
  {
    RRTNode* current_node = this;
    // RRTNode* current_child_node = NULL;
    RRTNode* new_node;
    RRTNode* new_child_node = NULL;

    while (current_node)
    {
      new_node = new RRTNode();
      new_node->state_ = current_node->state_;
      new_node->gain_ = current_node->gain_;
      new_node->gain_explicitly_calculated_ = current_node->gain_explicitly_calculated_;
      new_node->parent_ = NULL;

      if (new_child_node)
      {
        new_node->children_.push_back(new_child_node);
        new_child_node->parent_ = new_node;
      }

      // current_child_node = current_node;
      current_node = current_node->parent_;
      new_child_node = new_node;
    }

    return new_node;
  }


  // double score(double lambda, const Eigen::Vector4d& cur_state , double lambda_yaw = 10.0);

// palworld
  double score(double lambda, const Eigen::Vector4d& cur_state , double lambda_yaw = 0.0)
  {
    if (this->parent_){
      double parent_score = this->parent_->score(lambda, cur_state, lambda_yaw);
      double dis_exp = exp(-lambda * this->distance(this->parent_));
      double dis_yaw_exp = exp(-lambda_yaw * this->dis_yaw(this->parent_,cur_state));

      // std::cout << "----- score -----" << std::endl;
      // std::cout << "cur_state: " << cur_state.transpose() << std::endl;
      // std::cout << "this_state: " << this->state_.transpose() << std::endl;
      // std::cout << "dis_exp: " << dis_exp << std::endl;
      // std::cout << "dis_yaw_exp: " << dis_yaw_exp << std::endl;
      return parent_score + this->gain_ * dis_exp * dis_yaw_exp;
    }
    else
      return this->gain_;
  }

  double score(double lambda)
  {
    if (this->parent_)
      return this->parent_->score(lambda) +
             this->gain_ * exp(-lambda * this->distance(this->parent_));
            //  this->gain_ * exp(-lambda * this->cost());
    else
      return this->gain_;
  }

  double cost()
  {
    if (this->parent_)
      return this->distance(this->parent_) + this->parent_->cost();
    else
      return 0;
  }

  double distance(RRTNode* other)
  {
    Eigen::Vector3d p3(this->state_[0], this->state_[1], this->state_[2]);
    Eigen::Vector3d q3(other->state_[0], other->state_[1], other->state_[2]);
    return (p3 - q3).norm();
  }

  double dis_yaw(RRTNode* other, const Eigen::Vector4d& cur_state )
  {    double now_yaw = cur_state[3];

    Eigen::Vector2d dir2d((this->state_.head(2)-cur_state.head(2)).normalized());
    Eigen::Vector2d dir_now(cos(now_yaw), sin(now_yaw));

    // double this_yaw = this->state_[3];
    // double other_yaw = other->state_[3];
    // double other_yaw = state_[3];
    return -0.5*( dir2d.dot(dir_now) ) + 0.5;
  }

};

#endif
