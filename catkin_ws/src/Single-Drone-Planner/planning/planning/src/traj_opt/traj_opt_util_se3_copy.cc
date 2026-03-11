/*
 * @Author: BbbbigRui bbbbigrui@zju.edu.cn
 * @Date: 2023-03-02 19:15:00
 * @LastEditors: BbbbigRui bbbbigrui@zju.edu.cn
 * @LastEditTime: 2023-06-15 21:50:04
 * @FilePath: /src/planning/planning/src/traj_opt/traj_opt_util.cc
 * @Description: 
 * 
 * Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
 */
#include <traj_opt/traj_opt.h>
#include <traj_opt/lbfgs_raw.hpp>

namespace traj_opt {

//!!!!!!!!!!!!!!
std::pair<Eigen::VectorXd,Eigen::MatrixXd> TrajOpt::getPtsCostGradSE3(const Eigen::MatrixXd& pts,
                                                                      const Trajectory<5>& traj){
  
  int n_pts = pts.cols();
  Eigen::MatrixXd traj_gd(4*traj.getPieceNum()-3,n_pts);
  traj_gd.setZero();

  Eigen::VectorXd costP(n_pts);
  costP.setZero();

  // INFO_MSG("dis_gt:");

  for(int i=0;i<pts.cols();++i){

    Eigen::Vector3d pt_w=sdf_util_ptr_->ptNet2World(pts.col(i));

    auto pt_cost_gd = getPtCostGradSE3(pt_w,traj);
    costP(i) = rhoP_*pt_cost_gd.first;
    traj_gd.col(i) = rhoP_*pt_cost_gd.second;
  }

  return std::make_pair(costP,traj_gd);
}


/**
 * @description: 
 * @param pt   世界坐标系障碍物点
 * @param traj 世界坐标系轨迹
 * @return {*}
 */
// 解析
std::pair<double,Eigen::VectorXd> TrajOpt::getPtCostGradSE3(const Eigen::Vector3d& pt,
                                                            const Trajectory<5>& traj){

  double dist = sdf_util_ptr_->getSDF(pt,traj).first;

  // INFO_MSG("pt:"<<pt.transpose()<<",dist:"<<dist);

  // INFO_MSG("[getPtCostGradSE3]1");

  Eigen::VectorXd traj_gd(4*traj.getPieceNum()-3);
  traj_gd.setZero();

  double costP=0;

  sdfTrajGrad(pt,traj,traj_gd,costP);

  // INFO_MSG("[getPtCostGradSE3]2");

  return std::make_pair(costP,traj_gd);

}

void TrajOpt::vis_sdf(const std::vector<Eigen::Vector3d>& pts_vec,
              const Eigen::VectorXd& sdf_list,
              const Eigen::Vector3d& p0,
              const std::string& topic){

  std::vector<std::pair<Eigen::Vector3d,double>> pti_vec;

  for(int i=0;i<pts_vec.size();++i){
    pti_vec.push_back(std::make_pair(pts_vec[i]+p0,sdf_list(i)));
  }

  visPtr_->visualize_pointcloud_itensity(pti_vec,topic);
}

void TrajOpt::ZPlanePts(std::vector<Eigen::Vector3d>& pts_vec,
                        const double& z){

  pts_vec.clear();

  for(double x=-0.5*box_length_;x<=0.5*box_length_+1e-3;x+=box_res_){
    for(double y=-0.5*box_width_;y<=0.5*box_width_+1e-3;y+=box_res_){
    pts_vec.push_back(Eigen::Vector3d(x,y,z));
  }}
}


bool TrajOpt::trajTiltCheck(const Trajectory<5>& traj){
  double dur = traj.getTotalDuration();

  for(double t=0;t<=dur;t+=0.1){

    Eigen::Vector3d vel, acc, jer, snp;
    // double psi, dPsi, ddPsi 
    double thr;
    Eigen::Vector4d quat;
    Eigen::Vector3d bodyrate;

    vel = traj.getVel(t);
    acc = traj.getAcc(t);
    jer = traj.getJer(t);

    double cos_theta,theta;

    flatmap_.forward(vel, acc, jer, 0, 0, thr, quat, bodyrate);
    cos_theta = 1.0 - 2.0 * (quat(1) * quat(1) + quat(2) * quat(2));
    
    if(acos(cos_theta)>theta_limit_)
      return false;
  }

  return true;
}

/**
 * @description: 
 * @param pos_eva   障碍物
 * @param pos_ego   轨迹点
 * @param cost_sdf  SDF磨光后的cost
 * @return {*}
 */
void TrajOpt::grad_sdf_p(const double& sdf,
                          const Eigen::Vector3d& pos_eva,
                          const Eigen::Vector3d& pos_ego,
                          const Eigen::Vector4d& quat,
                          const Eigen::Vector3d gradp_eva,
                          const Eigen::Vector3d gradp_rel,
                          Eigen::Vector3d &gradp,
                          Eigen::Vector4d &grad_quat,
                          double& cost_sdf){

  cost_sdf = 0;
  grad_quat.setZero();
  gradp.setZero();

  Eigen::Vector3d p_minus_x;
  p_minus_x = pos_eva - pos_ego;

  double sdf_out_grad = 0.0;
  smoothedL1(dis_th_ - sdf, 0.01, cost_sdf, sdf_out_grad);

  Eigen::Vector3d sdf_grad;
  sdf_grad = sdf_out_grad * gradp_eva;

  if(cost_sdf>DBL_EPSILON){
    grad_quat(0) = gradp_rel.transpose() * sdf_util_ptr_->getQuatTransDW(quat) * p_minus_x;
    grad_quat(1) = gradp_rel.transpose() * sdf_util_ptr_->getQuatTransDX(quat) * p_minus_x;
    grad_quat(2) = gradp_rel.transpose() * sdf_util_ptr_->getQuatTransDY(quat) * p_minus_x;
    grad_quat(3) = gradp_rel.transpose() * sdf_util_ptr_->getQuatTransDZ(quat) * p_minus_x;
    // 正常版本
    grad_quat = -sdf_out_grad * grad_quat;
    gradp = sdf_grad;

    //和数值梯度做对比 不磨
    // grad_quat = -grad_quat;
    // gradp = gradp_eva;

    // INFO_MSG("[GSP]:pos:"<<pos_ego.transpose()<<"pos_eva:"<<pos_eva.transpose()
    //       <<"\n,cost_sdf:"<<cost_sdf<<",")
  }
}

//!!!!!!!!!!!!!!!!!!!!!!
// with smooth
/**
 * @description: 
 * @param pos_eva 障碍物世界系坐标
 * @param traj    世界系轨迹
 * @return {*}
 */
void TrajOpt::sdfTrajGrad(const Eigen::Vector3d& pos_eva,
                          const Trajectory<5>& traj,
                          Eigen::VectorXd& traj_gd,
                          double& cost_sdf){
  cost_sdf = 0;
  traj_gd.setZero();

  //!障碍物点的t_star,sdf,gradSDF; 飞机在t_star时的状态
  double sdf, t_star;
  Eigen::Vector3d pos, vel, acc, jer;
  Eigen::Vector3d bodyrate, gradSDF,gradSDF_rel;
  Eigen::Vector4d quat;
  cal_t_star_state(pos_eva,traj,sdf,t_star,
                    gradSDF,gradSDF_rel,pos, vel, acc,jer,quat,bodyrate);

  //!计算对t*点的gradP和gradQuat
  Eigen::Vector3d gradp;
  Eigen::Vector4d grad_quat;
  gradp.setZero();
  grad_quat.setZero();
  grad_sdf_p(sdf,pos_eva,pos,quat,gradSDF,gradSDF_rel,gradp,grad_quat,cost_sdf);
  // INFO_MSG("[sdfTrajGrad]gradp"<<gradp.transpose()<<",grad_quat:"<<grad_quat.transpose());
// ?这不是相对梯度 后面咋传到pvaj的

  if(cost_sdf<DBL_EPSILON)
    return;

  // INFO_MSG("pos_eva:"<<pos_eva.transpose()<<",cost_sdf:"<<cost_sdf<<",t_star:"<<t_star);

  //!梯度传到p,v,a,j
  Eigen::Vector3d gradPosTotal, gradVelTotal, gradAccTotal, gradJerTotal;
  double gradPsiT, gradDPsiT;
  gradPosTotal.setZero();
  gradVelTotal.setZero();
  gradAccTotal.setZero();
  gradJerTotal.setZero();
  gradPsiT = gradDPsiT = 0;
  sdf_util_ptr_->flat_backward(gradp,Eigen::Vector3d::Zero()/*vel*/,Eigen::Vector3d::Zero()/*acc*/,0./*thr*/,
                    grad_quat,Eigen::Vector3d::Zero()/*omg*/,
                    gradPosTotal,gradVelTotal,gradAccTotal,gradJerTotal,
                    gradPsiT,gradDPsiT);
  // INFO_MSG("[sdfTrajGrad]gradPosTotal:"<<gradPosTotal.transpose()<<",gradVelTotal:"<<gradVelTotal.transpose());
  // INFO_MSG("[sdfTrajGrad]gradAccTotal:"<<gradAccTotal.transpose()<<",gradJerTotal:"<<gradJerTotal.transpose());

  //! 计算gdT和gdP的梯度
  //输入gradViola_c,gradViola_t,i;输出gdP+gdT
  int N = traj.getPieceNum();
  Eigen::VectorXd gdT;
  Eigen::MatrixXd gdP;
  cal_gdT_gdP(traj,t_star,vel,acc,jer,
              gradPosTotal, gradVelTotal, gradAccTotal, gradJerTotal, 
              gdT, gdP);

  traj_gd.segment(0,N)=gdT;
  gdP.resize(3*(N - 1),1);
  traj_gd.segment(N,3*(N - 1))=gdP;
  // INFO_MSG("[sdfTrajGrad] 6");
  // INFO_MSG("解析      traj_gd:"<<traj_gd.transpose());

  return;
}

void TrajOpt::cal_t_star_state(const Eigen::Vector3d& pos_eva,
                                const Trajectory<5>& traj,
                                double& sdf,
                                double& t_star,
                                Eigen::Vector3d& gradSDF,
                                Eigen::Vector3d& gradSDF_rel,
                                Eigen::Vector3d& pos, 
                                Eigen::Vector3d& vel, 
                                Eigen::Vector3d& acc, 
                                Eigen::Vector3d& jer,
                                Eigen::Vector4d& quat,
                                Eigen::Vector3d& bodyrate){

  sdf = sdf_util_ptr_->getSDFAndGrad(pos_eva,traj,gradSDF,t_star);
  sdf_util_ptr_->cal_uav_state(t_star, traj, pos, vel, acc, jer, quat, bodyrate);
  Eigen::Quaterniond q_w2b(quat[0], -quat[1], -quat[2], -quat[3]);
  gradSDF_rel = q_w2b*gradSDF;
}

void TrajOpt::cal_gdT_gdP(const Trajectory<5>& traj,
                          const double& t_star,
                          const Eigen::Vector3d& vel, 
                          const Eigen::Vector3d& acc, 
                          const Eigen::Vector3d& jer,
                          const Eigen::Vector3d& gradPosTotal, 
                          const Eigen::Vector3d& gradVelTotal, 
                          const Eigen::Vector3d& gradAccTotal, 
                          const Eigen::Vector3d& gradJerTotal,
                          Eigen::VectorXd& gdT,
                          Eigen::MatrixXd& gdP){

    //!梯度传到C和T
    double t_star_temp = t_star;
    int i = traj.locatePieceIdx(t_star_temp);
    Eigen::Matrix<double, 6, 1> beta0, beta1, beta2, beta3;

    // 从 Trajectory 恢复 MINCO
    int N = traj.getPieceNum();
    Eigen::MatrixXd P = traj.getPositions().block(0,1,3,N-1);
    Eigen::VectorXd T = traj.getDurations();
    Eigen::Matrix3d inState,outState;
    traj.getInOutStates(inState,outState);

    minco::MINCO_S3 minco_s3_opt;
    minco_s3_opt.reset(inState,outState,N);
    minco_s3_opt.generate(P,T);

    beta0 = cal_timebase_jerk(0, t_star_temp);
    beta1 = cal_timebase_jerk(1, t_star_temp);
    beta2 = cal_timebase_jerk(2, t_star_temp);
    beta3 = cal_timebase_jerk(3, t_star_temp);
    // INFO_MSG("[sdfTrajGrad] 4.5");

    Eigen::Matrix<double, 6, 3> gradViola_c;
    double gradViola_t;
    gradViola_c = beta0 * gradPosTotal.transpose();
    gradViola_t = gradPosTotal.transpose() * vel;
    gradViola_c += beta1 * gradVelTotal.transpose();
    gradViola_t += gradVelTotal.transpose() * acc;
    gradViola_c += beta2 * gradAccTotal.transpose();
    gradViola_t += gradAccTotal.transpose() * jer;
    gradViola_c += beta3 * gradJerTotal.transpose();
    // INFO_MSG("[sdfTrajGrad] 5");

    //! 计算gdT和gdP的梯度
    //输入gradViola_c,gradViola_t,i;输出gdP+gdT
    gdT.resize(N);
    gdP.resize(3, N - 1);
    gdT.setZero();
    gdP.setZero();

    Eigen::MatrixXd gdC(8 * N, 3);
    gdC.setZero();
    gdC.block<6, 3>(i * 6, 0) = gradViola_c;

    for(int j=0;j<i;++j){
      gdT(j)-=gradViola_t;
    }

    minco_s3_opt.calGradPT(gdC,gdT,gdP);
}






/**
 * @description: 
 * @param pos_eva   障碍物
 * @param pos_ego   轨迹点
 * @param cost_sdf  SDF磨光后的cost
 * @return {*}
 */
void TrajOpt::grad_sdf_p_no_smooth(const double& sdf,
                          const Eigen::Vector3d& pos_eva,
                          const Eigen::Vector3d& pos_ego,
                          const Eigen::Vector4d& quat,
                          const Eigen::Vector3d gradp_eva,
                          const Eigen::Vector3d gradp_rel,
                          Eigen::Vector3d &gradp,
                          Eigen::Vector4d &grad_quat){

  grad_quat.setZero();
  gradp.setZero();

  Eigen::Vector3d p_minus_x;
  p_minus_x = pos_eva - pos_ego;

  grad_quat(0) = gradp_rel.transpose() * sdf_util_ptr_->getQuatTransDW(quat) * p_minus_x;
  grad_quat(1) = gradp_rel.transpose() * sdf_util_ptr_->getQuatTransDX(quat) * p_minus_x;
  grad_quat(2) = gradp_rel.transpose() * sdf_util_ptr_->getQuatTransDY(quat) * p_minus_x;
  grad_quat(3) = gradp_rel.transpose() * sdf_util_ptr_->getQuatTransDZ(quat) * p_minus_x;
  
  grad_quat = grad_quat;
  gradp = -gradp_eva;

}



/**
 * @description: 
 * @param pos_eva 障碍物世界系坐标
 * @param traj    世界系轨迹
 * @return {*}
 */
void TrajOpt::sdfTrajGrad(const Eigen::Vector3d& pos_eva,
                          const Trajectory<5>& traj,
                          Eigen::VectorXd& traj_gd){

  int N = traj.getPieceNum();
  traj_gd.resize(4*N-3);
  traj_gd.setZero();

  //!障碍物点的t_star,sdf,gradSDF; 飞机在t_star时的状态
  double sdf, t_star;
  Eigen::Vector3d pos, vel, acc, jer;
  Eigen::Vector3d bodyrate, gradSDF,gradSDF_rel;
  Eigen::Vector4d quat;
  cal_t_star_state(pos_eva,traj,sdf,t_star,
                    gradSDF,gradSDF_rel,pos, vel, acc,jer,quat,bodyrate);

  //!计算对t*点的gradP和gradQuat
  Eigen::Vector3d gradp;
  Eigen::Vector4d grad_quat;
  gradp.setZero();
  grad_quat.setZero();
  grad_sdf_p_no_smooth(sdf,pos_eva,pos,quat,gradSDF,gradSDF_rel,gradp,grad_quat);

  //!梯度传到p,v,a,j
  Eigen::Vector3d gradPosTotal, gradVelTotal, gradAccTotal, gradJerTotal;
  double gradPsiT, gradDPsiT;
  gradPosTotal.setZero();
  gradVelTotal.setZero();
  gradAccTotal.setZero();
  gradJerTotal.setZero();
  gradPsiT = gradDPsiT = 0;
  sdf_util_ptr_->flat_backward(gradp,Eigen::Vector3d::Zero()/*vel*/,Eigen::Vector3d::Zero()/*acc*/,0./*thr*/,
                    grad_quat,Eigen::Vector3d::Zero()/*omg*/,
                    gradPosTotal,gradVelTotal,gradAccTotal,gradJerTotal,
                    gradPsiT,gradDPsiT);
  // INFO_MSG("[sdfTrajGrad]gradPosTotal:"<<gradPosTotal.transpose()<<",gradVelTotal:"<<gradVelTotal.transpose());
  // INFO_MSG("[sdfTrajGrad]gradAccTotal:"<<gradAccTotal.transpose()<<",gradJerTotal:"<<gradJerTotal.transpose());


  //! 梯度传到T,P
  Eigen::VectorXd gdT;
  Eigen::MatrixXd gdP;
  cal_gdT_gdP(traj,t_star,vel,acc,jer,
              gradPosTotal, gradVelTotal, gradAccTotal, gradJerTotal, 
              gdT, gdP);

  traj_gd.segment(0,N)=gdT;
  gdP.resize(3*(N - 1),1);
  traj_gd.segment(N,3*(N - 1))=gdP;
  // INFO_MSG("[sdfTrajGrad] 6");
  // INFO_MSG("解析      traj_gd:"<<traj_gd.transpose());
  // traj_gd = sdf_util_ptr_->calTrajGrad(pos_eva,traj,5e-8);
  // INFO_MSG("数值 5e-8 traj_gd:"<<traj_gd.transpose());
  // traj_gd = sdf_util_ptr_->calTrajGrad(pos_eva,traj,1e-7);
  // INFO_MSG("数值 1e-7 traj_gd:"<<traj_gd.transpose());
  // traj_gd = sdf_util_ptr_->calTrajGrad(pos_eva,traj,5e-7);
  // INFO_MSG("数值 5e-7 traj_gd:"<<traj_gd.transpose());
  // traj_gd = sdf_util_ptr_->calTrajGrad(pos_eva,traj,1e-6);
  // INFO_MSG("数值 1e-6 traj_gd:"<<traj_gd.transpose());
  // traj_gd = sdf_util_ptr_->calTrajGrad(pos_eva,traj,5e-6);
  // INFO_MSG("数值 5e-6 traj_gd:"<<traj_gd.transpose());
  // traj_gd = sdf_util_ptr_->calTrajGrad(pos_eva,traj,1e-5);
  // INFO_MSG("数值 1e-5 traj_gd:"<<traj_gd.transpose());
  // traj_gd = sdf_util_ptr_->calTrajGrad(pos_eva,traj,5e-5);
  // INFO_MSG("数值 5e-5 traj_gd:"<<traj_gd.transpose());
  // traj_gd = sdf_util_ptr_->calTrajGrad(pos_eva,traj,1e-4);
  // INFO_MSG("数值 1e-4 traj_gd:"<<traj_gd.transpose());
  // traj_gd = sdf_util_ptr_->calTrajGrad(pos_eva,traj,5e-4);
  // INFO_MSG("数值 5e-4 traj_gd:"<<traj_gd.transpose());
  // INFO_MSG("");
  // INFO_MSG("");

  return;
}


/**
 * @description: 
 * @param pos_eva 障碍物世界系坐标
 * @param traj    世界系轨迹
 * @return {*}
 */
// 返回的比较多
void TrajOpt::sdfTrajGrad(const Eigen::Vector3d& pos_eva,
                          const Trajectory<5>& traj,
                          const double& t_star,
                          const double& sdf,
                          const Eigen::Vector3d& gradSDF,
                          Eigen::VectorXd& traj_gd){

  int N = traj.getPieceNum();
  traj_gd.resize(4*N-3);
  traj_gd.setZero();

  //!障碍物点的t_star,sdf,gradSDF; 飞机在t_star时的状态
  Eigen::Vector3d pos, vel, acc, jer;
  Eigen::Vector3d bodyrate, gradSDF_rel;
  Eigen::Vector4d quat;
  sdf_util_ptr_->cal_uav_state(t_star, traj, pos, vel, acc, jer, quat, bodyrate);
  Eigen::Quaterniond q_w2b(quat[0], -quat[1], -quat[2], -quat[3]);
  gradSDF_rel = q_w2b*gradSDF;

  //!计算对t*点的gradP和gradQuat
  Eigen::Vector3d gradp;
  Eigen::Vector4d grad_quat;
  gradp.setZero();
  grad_quat.setZero();
  grad_sdf_p_no_smooth(sdf,pos_eva,pos,quat,gradSDF,gradSDF_rel,gradp,grad_quat);

  //!梯度传到p,v,a,j
  Eigen::Vector3d gradPosTotal, gradVelTotal, gradAccTotal, gradJerTotal;
  double gradPsiT, gradDPsiT;
  gradPosTotal.setZero();
  gradVelTotal.setZero();
  gradAccTotal.setZero();
  gradJerTotal.setZero();
  gradPsiT = gradDPsiT = 0;
  sdf_util_ptr_->flat_backward(gradp,Eigen::Vector3d::Zero()/*vel*/,Eigen::Vector3d::Zero()/*acc*/,0./*thr*/,
                    grad_quat,Eigen::Vector3d::Zero()/*omg*/,
                    gradPosTotal,gradVelTotal,gradAccTotal,gradJerTotal,
                    gradPsiT,gradDPsiT);

  //! 梯度传到T,P
  Eigen::VectorXd gdT;
  Eigen::MatrixXd gdP;
  cal_gdT_gdP(traj,t_star,vel,acc,jer,
              gradPosTotal, gradVelTotal, gradAccTotal, gradJerTotal, 
              gdT, gdP);

  traj_gd.segment(0,N)=gdT;
  gdP.resize(3*(N - 1),1);
  traj_gd.segment(N,3*(N - 1))=gdP;

  return;
}

// Input: p_obs, traj
// Output: cost_sdf, grad_traj
void TrajOpt::sdfTrajGrad(const Eigen::Vector3d& p_obs,
                          const Trajectory<5>& traj,
                          const double& sdf,
                          const Eigen::Vector3d& gradSDF,
                          Eigen::VectorXd& traj_gd){
  

}



} // namespace traj_opts