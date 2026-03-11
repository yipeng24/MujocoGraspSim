#ifndef SDF_UTIL_HPP
#define SDF_UTIL_HPP

#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <iostream>
#include <vector>


#include "util_gym/util_gym.hpp"
#include "minco.hpp"
#include "flatness.hpp"
#include <quadrotor_msgs/PolyTraj.h>

#define ORDER_5
#define TRAJ_ORDER 5

// #define ORDER_7
// #define TRAJ_ORDER 7
namespace sdf_util{

class SDFUtil{

private:

  Eigen::Matrix4d Trans_;
  // Eigen::Map<Eigen::Matrix3d> Rot_(Trans_.data(),3,3);
  // Eigen::Map<Eigen::Vector3d> Tr_(Trans_.data()+12,3);
  // Eigen::Map<double> scale_(Trans_.data()+15,1);

  // flatness map
  flatness::FlatnessMap flatmap_;

  double sdf_min_;//其实也类似与骨架厚度

  std::vector<std::pair<Eigen::Vector3d,Eigen::Vector3d>> lines_;//机体骨架
  std::vector<std::pair<Eigen::Vector4d,Eigen::Vector3d>> cakes_;//机体骨架

  double alpha_guide_;//渐进引导权重

public:

  SDFUtil(){}
  ~SDFUtil() {}

  void setSDFMin(const double& sdf_min){sdf_min_=sdf_min;}

  void setSkeleton(const std::vector<std::pair<Eigen::Vector3d,Eigen::Vector3d>>& lines,
                   const std::vector<std::pair<Eigen::Vector4d,Eigen::Vector3d>>& cakes){
    lines_=lines;
    cakes_=cakes;
  }

  void set_alpha_guide(const double& alpha){
    alpha_guide_=alpha;
  }

  void setFlatMap(const flatness::FlatnessMap& flatmap){flatmap_=flatmap;}

  void calTrans(const Trajectory<TRAJ_ORDER>& traj){

    double dur = traj.getTotalDuration();

    Eigen::Matrix3d Rot;
    Eigen::Vector3d T;

    Eigen::Vector3d iniP(traj.getPos(0)),finP(traj.getPos(dur));
    T=-0.5*(iniP+finP);

    double scale = (iniP-finP).norm();

    // INFO_MSG("iniP:"<<iniP.transpose()<<",finP:"<<finP.transpose());

    Eigen::Vector3d x1 = (finP-iniP).normalized();

    Eigen::Matrix2d R_z_90;
    R_z_90<<0,-1,1,0;

    Eigen::Vector2d y1_xy = R_z_90*x1.segment(0,2).normalized();
    Eigen::Vector3d y1; y1(0)=y1_xy(0); y1(1)=y1_xy(1); y1(2)=0;
    Eigen::Vector3d z1=x1.cross(y1);

    // INFO_MSG("x1:"<<x1.transpose()<<",y1:"<<y1.transpose()<<",z1:"<<z1.transpose());

    // Eigen::Matrix3d Rot;
    Rot.row(0)=x1.transpose();
    Rot.row(1)=y1.transpose();
    Rot.row(2)=z1.transpose();

    Trans_.setZero();
    Trans_.block(0,0,3,3) = Rot;
    //cpp中的norm_traj起点0-1
    Trans_.block(0,3,3,1) = T;
    Trans_(3,3) = scale;
  }

  Eigen::Matrix4d getTrans(){return Trans_;}
  double getScale(){return Trans_(3,3);}

  Eigen::Vector3d ptNet2World(const Eigen::Vector3d& pt_n){
    Eigen::Matrix3d R_w2n = Trans_.block(0,0,3,3);
    Eigen::Vector3d Tr = Trans_.block(0,3,3,1);//Transfer
    double scale = Trans_(3,3);

    return scale*R_w2n.transpose()*pt_n-Tr;
  }

  Eigen::Vector3d ptWorld2Net(const Eigen::Vector3d& pt_w){
    Eigen::Matrix3d R_w2n = Trans_.block(0,0,3,3);
    Eigen::Vector3d Tr = Trans_.block(0,3,3,1);//Transfer
    double scale_inv = 1.0/Trans_(3,3);

    return scale_inv*R_w2n*(pt_w+Tr);
  }

  Eigen::Vector3d ptWorld2NetNoScale(const Eigen::Vector3d& pt_w){
    Eigen::Matrix3d R_w2n = Trans_.block(0,0,3,3);
    Eigen::Vector3d Tr = Trans_.block(0,3,3,1);//Transfer

    return R_w2n*(pt_w+Tr);
  }

  Eigen::Vector3d ptWorld2Net_Rot(const Eigen::Vector3d& pt_w){
    Eigen::Matrix3d R_w2n = Trans_.block(0,0,3,3);
    return R_w2n*pt_w;
  }


  bool generate_traj(const Eigen::MatrixXd& ini_v_a,
                      const Eigen::Matrix3d& finState,
                      const Eigen::VectorXd& T,
                      const Eigen::MatrixXd& P,
                      Trajectory<TRAJ_ORDER>& traj){

    int N_ = P.cols() + 1;

    Eigen::Matrix3d iniState;
    iniState.setZero();
    iniState(0,0)=-0.5;
    iniState.block(0,1,3,2)=ini_v_a;

    // std::cout << "ini_v_a\n" << ini_v_a << std::endl;
    // std::cout << "iniState\n" << iniState << std::endl;
    // std::cout << "finState\n" << finState << std::endl;
    // std::cout << "P:\n" << P << std::endl;
    // std::cout << "T:" << T.transpose() << std::endl;  

    minco::MINCO_S3 minco_s3_opt;

    minco_s3_opt.reset(iniState.block<3,3>(0,0), finState.block<3,3>(0,0), N_);
    minco_s3_opt.generate(P, T);
    traj = minco_s3_opt.getTraj();

    return true;
  }

  bool generate_traj2(const Eigen::Matrix3d& iniState,
                    const Eigen::Matrix3d& finState,
                    const Eigen::VectorXd& T,
                    const Eigen::MatrixXd& P,
                    Trajectory<TRAJ_ORDER>& traj){

    int N_ = P.cols() + 1;

    // std::cout << "ini_v_a\n" << ini_v_a << std::endl;
    // std::cout << "iniState\n" << iniState << std::endl;
    // std::cout << "finState\n" << finState << std::endl;
    // std::cout << "P:\n" << P << std::endl;
    // std::cout << "T:" << T.transpose() << std::endl;  

    minco::MINCO_S3 minco_s3_opt;

    minco_s3_opt.reset(iniState.block<3,3>(0,0), finState.block<3,3>(0,0), N_);
    minco_s3_opt.generate(P, T);
    traj = minco_s3_opt.getTraj();

    return true;
  }



  Eigen::VectorXd traj2trajInfoNorm(const Trajectory<TRAJ_ORDER>& traj){
    int n_pie = traj.getPieceNum();
    Eigen::VectorXd traj_info(4*n_pie+3);
    traj_info.setZero();

    Eigen::Map<Eigen::MatrixXd> ini_v_a(traj_info.data(),3,2);
    Eigen::Map<Eigen::VectorXd> T(traj_info.data()+6,n_pie);
    Eigen::Map<Eigen::MatrixXd> P(traj_info.data()+6+n_pie,3,n_pie-1);

    T = traj.getDurations();
    P = traj.getPositions().block(0,1,3,n_pie-1);

    for(int i=0;i<n_pie-1;++i)
      P.col(i)=ptWorld2Net(P.col(0));

    return traj_info;
  }

  Eigen::VectorXd traj2trajInfoNorm(const Trajectory<TRAJ_ORDER>& traj,
                                    Eigen::MatrixXd& ini_v_a){
    int n_pie = traj.getPieceNum();
    Eigen::VectorXd traj_info(4*n_pie-3);
    traj_info.setZero();

    ini_v_a.resize(3,2);
    ini_v_a.col(0) = traj.getVel(0);
    ini_v_a.col(1) = traj.getAcc(0);

    Eigen::Map<Eigen::VectorXd> T(traj_info.data(),n_pie);
    Eigen::Map<Eigen::MatrixXd> P(traj_info.data()+n_pie,3,n_pie-1);

    T = traj.getDurations();
    P = traj.getPositions().block(0,1,3,n_pie-1);

    for(int i=0;i<n_pie-1;++i)
      P.col(i)=ptWorld2Net(P.col(i));

    return traj_info;
  }

  Eigen::MatrixXd Pworld2net(const Eigen::MatrixXd& P){
    Eigen::MatrixXd P_net(P);
    for(int i=0;i<P.cols();++i)
      P_net.col(i) = ptWorld2Net(P.col(i));
    return P_net;
  }

  Trajectory<TRAJ_ORDER> trajInfo2traj(const Eigen::VectorXd& traj_info){

    int n_pie = (traj_info.rows()-3)/4;

  // INFO_MSG("traj_info.rows():"<<traj_info.rows()<<",n_pie:"<<n_pie);

    Eigen::VectorXd traj_info1(traj_info);

    Eigen::Map<Eigen::MatrixXd> ini_v_a(traj_info1.data(),3,2);
    Eigen::Map<Eigen::VectorXd> T(traj_info1.data()+6,n_pie);
    Eigen::Map<Eigen::MatrixXd> P(traj_info1.data()+6+n_pie,3,n_pie-1);

    Eigen::Matrix3d finState;
    finState.setZero();
    finState.col(0)=Eigen::Vector3d(0.5,0,0);

    // INFO_MSG_BLUE("[trajInfo2traj]ini_v_a:\n"<<ini_v_a);
    // INFO_MSG_BLUE("[trajInfo2traj]finState:\n"<<finState);
    // INFO_MSG_BLUE("[trajInfo2traj]T:"<<T);
    // INFO_MSG_BLUE("[trajInfo2traj]P:\n"<<P.transpose());

    Trajectory<TRAJ_ORDER> traj;
    bool suc = generate_traj(ini_v_a,finState,T,P,traj);

    return traj;
  }


  Trajectory<TRAJ_ORDER> trajInfo2traj(const Eigen::VectorXd& traj_info,
                              const Eigen::MatrixXd& ini_v_a){

    int n_pie = (traj_info.rows()+3)/4;

  // INFO_MSG("traj_info.rows():"<<traj_info.rows()<<",n_pie:"<<n_pie);

    Eigen::VectorXd traj_info1(traj_info);

    Eigen::Map<Eigen::VectorXd> T(traj_info1.data(),n_pie);
    Eigen::Map<Eigen::MatrixXd> P(traj_info1.data()+n_pie,3,n_pie-1);

    Eigen::Matrix3d finState;
    finState.setZero();
    finState.col(0)=Eigen::Vector3d(0.5,0,0);

    // INFO_MSG_BLUE("[trajInfo2traj]ini_v_a:\n"<<ini_v_a);
    // INFO_MSG_BLUE("[trajInfo2traj]finState:\n"<<finState);
    // INFO_MSG_BLUE("[trajInfo2traj]T:"<<T);
    // INFO_MSG_BLUE("[trajInfo2traj]P:\n"<<P.transpose());

    Trajectory<TRAJ_ORDER> traj;
    bool suc = generate_traj(ini_v_a,finState,T,P,traj);

    return traj;
  }

  //TODO
  bool trajNormalize(const Trajectory<TRAJ_ORDER>& traj0,
                      Trajectory<TRAJ_ORDER>& traj){

    Eigen::MatrixXd ini_v_a(3,2);ini_v_a.setZero();
    Eigen::VectorXd traj_info = traj2trajInfoNorm(traj0,ini_v_a);
    
    traj = trajInfo2traj(traj_info,ini_v_a);

    return true;
  }


  double smoothedL1(const double& x,
                        double& grad) {
    static double mu = 0.04;
    if (x < 0.0) {
      return 0.0;
    } else if (x > mu) {
      grad = 1.0;
      return x;
    } else {
      const double xdmu = x / mu;
      const double sqrxdmu = xdmu * xdmu;
      const double mumxd2 = mu - 0.5 * x;
      grad = sqrxdmu * ((-0.5) * xdmu + 3.0 * mumxd2 / mu);
      return mumxd2 * sqrxdmu * xdmu + 0.5 * mu;;
    }
  }

  Eigen::MatrixXd cal_timebase_snap(const int order, const double& t){
    double s1, s2, s3, s4, s5, s6, s7;
    s1 = t;
    s2 = s1 * s1;
    s3 = s2 * s1;
    s4 = s2 * s2;
    s5 = s4 * s1;
    s6 = s4 * s2;
    s7 = s4 * s3;
    Eigen::Matrix<double, 8, 1> beta;
    switch(order){
      case 0:
        beta << 1.0, s1, s2, s3, s4, s5, s6, s7;
        break;
      case 1:
        beta << 0.0, 1.0, 2.0 * s1, 3.0 * s2, 4.0 * s3, 5.0 * s4, 6.0 * s5, 7.0 * s6;
        break;
      case 2:
        beta << 0.0, 0.0, 2.0, 6.0 * s1, 12.0 * s2, 20.0 * s3, 30.0 * s4, 42.0 * s5;
        break;
      case 3:
        beta << 0.0, 0.0, 0.0, 6.0, 24.0 * s1, 60.0 * s2, 120.0 * s3, 210.0 * s4;
        break;
      case 4:
        beta << 0.0, 0.0, 0.0, 0.0, 24.0, 120.0 * s1, 360.0 * s2, 840.0 * s3;
        break;
      default:
        std::cout << "[trajopt] cal_timebase error." << std::endl;
        break;
    }
    return beta;
  }

  void calQuat(const double& t,
              const Trajectory<TRAJ_ORDER>& traj,
              Eigen::Quaterniond& q){

      Eigen::Vector3d vel, acc, jer, snp;
      // double psi, dPsi, ddPsi, thr;
      double thr;
      Eigen::Vector4d quat;
      Eigen::Vector3d bodyrate;


      vel = traj.getVel(t);
      acc = traj.getAcc(t);
      jer = traj.getJer(t);

      flatmap_.forward(vel, acc, jer, 0, 0, thr, quat, bodyrate);

      q = Eigen::Quaterniond(quat[0], quat[1], quat[2], quat[3]);
  }

  void calRot(const double& t,
              const Trajectory<TRAJ_ORDER>& traj,
              Eigen::Matrix3d& Rot){

    Eigen::Quaterniond quat;
    calQuat(t,traj,quat);

    Rot = quat.toRotationMatrix();
    // INFO_MSG("[calRot]scale:"<<scale<<",vel:"<<vel.transpose()<<",acc:"<<acc.transpose());
    // INFO_MSG("[calRot]Rot\n"<<Rot);
  }

  double disPt2Cake(const Eigen::Vector3d& p, //查询点
                    const Eigen::Vector4d& o_r, //中心点和半径
                    const Eigen::Vector3d& n){
    // h       p
    // |-------
    // |     /
    // |    / 
    // |   /
    // |n /     //注意n是单位向量
    // |-/theta
    // |/      r
    // .------------------|
    // o

    Eigen::Vector3d o=o_r.segment(0,3);
    double r=o_r(3);

    Eigen::Vector3d op(p-o);
    double cos,sin;
    cos = fabs(op.dot(n)/(op.norm()*n.norm()));
    sin = sqrt(1-cos*cos);

    double l_oh,l_hp;
    l_oh=op.norm()*cos;
    l_hp=op.norm()*sin;

    if(l_hp<=r)//垂足 in cake
      return l_oh;
    else
      return sqrt((l_hp-r)*(l_hp-r)+l_oh*l_oh);
  }

  double disPt2Line(const Eigen::Vector3d& p,//查询点
                      const Eigen::Vector3d& a, //线段端点
                      const Eigen::Vector3d& b){
    // h   a           b
    // ....------------
    // |  /
    // | /
    // |/
    // .p

    // INFO_MSG("p:"<<p.transpose()<<",a:"<<a.transpose()<<",b:"<<b.transpose());

    // method 1 通过垂足位置判断
    // /*
    Eigen::Vector3d ap(p-a),ab(b-a),bp(p-b);

    // r = |ap|*cos/|ab|
    double r=ap.dot(ab)/((ab.norm()*ab.norm())+1e-6);//垂足归一化

    if(r>0 && r<1){//垂足在线段内
      double cos=ap.dot(ab)/((ap.norm()*ab.norm())+1e-6);
      return ap.norm()*sqrt(1-cos*cos);
    }
    else if (r<=0)
      return ap.norm();
    else
      return bp.norm();
    // */

    // if(p.z()>a.z())
    //   return (p-a).norm();
    // else if(p.z()<b.z())
    //   return (p-b).norm();
    // else
    //   return Eigen::Vector2d(p.x(),p.y()).norm();
  }


  // #define GUIDE_TRAIN #渐进引导训练作图


  double getSDF_body(const Eigen::Vector3d& p){

    double dis=999;

    // 质点
    // return p.norm();

    dis = dist2cakeAndline(p);


    #ifdef GUIDE_TRAIN
      dis = alpha_guide_*dist2cakeAndline(p)+(1-alpha_guide_)*p.norm();
    #endif


    return dis+sdf_min_;
  }

  double dist2cakeAndline(const Eigen::Vector3d& p)
  {
    double dis=999;
    for(int i=0;i<(int)lines_.size();++i){
      // std::cout<<i<<":";
      double dis_pt2line = disPt2Line(p,lines_[i].first,lines_[i].second);
      dis=dis_pt2line<dis?dis_pt2line:dis;
      // std::cout<<"dis_pt2line:"<<dis_pt2line<<std::endl;
    }

    for(int i=0;i<(int)cakes_.size();++i){
      // std::cout<<i<<":";
      double dis_pt2cake = disPt2Cake(p,cakes_[i].first,cakes_[i].second);
      dis=dis_pt2cake<dis?dis_pt2cake:dis;
      // std::cout<<"dis_pt2cake:"<<dis_pt2cake<<std::endl;
    }
    return dis;
  }

  double getSDF_body_flt(const Eigen::Vector3d& pt){

    double dis = 0.;

    double resolution = 1e-2;
    for (int x = 0; x < 2; x++) {
      for (int y = 0; y < 2; y++) {
        for (int z = 0; z < 2; z++) {
            Eigen::Vector3d p;
            p = 2.0*Eigen::Vector3d(x,y,z) - Eigen::Vector3d::Ones();
            p = pt + resolution * p;
            dis += getSDF_body(p);
    }}}

    return dis*0.125;
  }
  

  double getDis(const Eigen::Vector3d& p, 
                const double& t, 
                const Trajectory<TRAJ_ORDER>& traj,
                const Eigen::Vector3d v=Eigen::Vector3d::Zero()){

    double dis;
    
    // Particle_Model
    // dis = (p-traj.getPos(t)).norm()+sdf_min_;

    // Non_Particle_Model
    Eigen::Matrix3d R_b2w;
    calRot(t,traj,R_b2w);
    Eigen::Vector3d p_b = R_b2w.transpose()*(p-traj.getPos(t));
    dis = getSDF_body(p_b);


    return dis;

    // double grad;
    // return smoothedL1(dis-sdf_min_,grad)+sdf_min_;
  }

// t*的梯度 参考ztr论文
double getGradT(const Eigen::Vector3d& p_ob, const double& t, const Trajectory<TRAJ_ORDER>& traj){
  Eigen::Vector3d pos, vel, acc, jer;
  Eigen::Vector4d quat;
  Eigen::Vector3d bodyrate;

  cal_uav_state(t,traj, pos, vel, acc, jer, quat, bodyrate);

  Eigen::Matrix3d R_b2w = Eigen::Quaterniond(quat[0], quat[1], quat[2], quat[3]).toRotationMatrix();
  Eigen::Vector3d x_rel = R_b2w.transpose()*(p_ob-pos);

  Eigen::Matrix3d omg_bar;
  omg_bar << 0, -bodyrate(2), bodyrate(1),
              bodyrate(2), 0, -bodyrate(0),
            -bodyrate(1),  bodyrate(0), 0;

  return getSDFGradBody(x_rel).transpose() //SDF x_rel的梯度
          *(omg_bar*R_b2w.transpose()*(pos-p_ob)
          -R_b2w.transpose()*vel);
}

void cal_uav_state(const double& t, const Trajectory<TRAJ_ORDER>& traj,
                   Eigen::Vector3d& pos, Eigen::Vector3d& vel, Eigen::Vector3d& acc, Eigen::Vector3d& jer,
                   Eigen::Vector4d& quat, Eigen::Vector3d& bodyrate){

    // Eigen::Vector3d pos, vel, acc, jer, snp;
    // double psi, dPsi, ddPsi, thr;
    double thr;

    pos = traj.getPos(t);
    vel = traj.getVel(t);
    acc = traj.getAcc(t);
    jer = traj.getJer(t);

    flatmap_.forward(vel, acc, jer, 0, 0, thr, quat, bodyrate);
} 

void flat_backward(const Eigen::Vector3d &pos_grad,
                    const Eigen::Vector3d &vel_grad,
                    const Eigen::Vector3d &acc_grad,
                    const double &thr_grad,
                    const Eigen::Vector4d &quat_grad,
                    const Eigen::Vector3d &omg_grad,
                    Eigen::Vector3d &pos_total_grad,
                    Eigen::Vector3d &vel_total_grad,
                    Eigen::Vector3d &acc_total_grad,
                    Eigen::Vector3d &jer_total_grad,
                    double &psi_total_grad,
                    double &dpsi_total_grad){

    flatmap_.backward(pos_grad, vel_grad, acc_grad, thr_grad,quat_grad, omg_grad,
                      pos_total_grad, vel_total_grad, acc_total_grad, jer_total_grad,
                      psi_total_grad,dpsi_total_grad);
} 


  double getIntStepScale(const double& t,const Trajectory<TRAJ_ORDER>& traj){

    Eigen::Vector3d pos, vel, acc, jer, snp;
    // double psi, dPsi, ddPsi, thr;
    double thr;
    Eigen::Vector4d quat;
    Eigen::Vector3d bodyrate;

    pos = traj.getPos(t);
    vel = traj.getVel(t);
    acc = traj.getAcc(t);
    jer = traj.getJer(t);

    flatmap_.forward(vel, acc, jer, 0, 0, thr, quat, bodyrate);

    // double 
    // bodyrate.squaredNorm() - omega_max*omega_max ;

    return 1;
  }


  std::pair<double,double> getSDF(const Eigen::Vector3d& p, const Trajectory<TRAJ_ORDER>& traj,const double t_seed0 = -1){
      double duration = traj.getTotalDuration();
      double sdf = 9999;
      double t_star,t_seed;

      // INFO_MSG_RED("getSDF");

      double int_step=1e-2;
      if(t_seed0 < 0){
        // get t_seed

        // double int_step=duration*1e-3;
        for (double t = 0; t < duration; t += int_step) {
          double dis = getDis(p,t,traj);

          if(dis < sdf){
            sdf = dis;
            t_seed = t;
          }
        }
      }
      else{
        sdf = getDis(p,t_seed0,traj);
        t_seed = t_seed0;
      }

      // INFO_MSG_RED("t_star0:"<<t_star<<",sdf:"<<sdf);

      // 梯度下降
      // #ifdef GRAD_SEARCH
      std::function<double(const double)> SDF = [&](const double t) -> double
      {
        return getDis(p,t,traj);
      };

      std::function<double(const double)> SDF_DOT = [&](const double t) -> double
      {
        return getGradT(p,t,traj);
      };

      double t_min = t_seed - int_step >= 0 ? t_seed - int_step : 0;
      double t_max = t_seed + int_step <= duration ? t_seed + int_step : duration;

      // INFO_MSG_BLUE("t_seed:"<<t_seed<<",t_min:"<<t_min<<",t_max:"<<t_max);
      gradient_descent(0, t_min, t_max, SDF, SDF_DOT, t_seed, sdf, t_star);
      // #endif


      // 二分法 假设是凸的
      #ifdef BIN_SEARCH
      // start search when t belong (1e-3,duration-1e-3)
      double err = 1e-5;
      // double d_t_star = getGradT(p,t_star,traj);

      double t0,t1;
      // 没有考虑t0或者t1为边界
      // if(d_t_star>0){
      //   t0 = t_star - int_step >= 0 ? t_star - int_step : 0;
      //   t1 = t_star;
      // }
      // else{
      //   t0 = t_star;
      //   t1 = t_star + int_step <= duration ? t_star + int_step : duration;
      // }
      t0 = t_seed - int_step >= 0 ? t_seed - int_step : 0;
      t1 = t_seed + int_step <= duration ? t_seed + int_step : duration;

      bool less_flag, end_search = false;
      int cnt = 1;
      while(!end_search){
        double dis0 = getDis(p,t0,traj);
        double dis1 = getDis(p,t1,traj);

        // INFO_MSG_RED("iter:"<<cnt);
        // INFO_MSG("t0:"<< std::setprecision(10)<<t0<<",t1:"<< std::setprecision(10)<<t1);
        // INFO_MSG("dis0:"<< std::setprecision(10)<<dis0<<",dis1:"<< std::setprecision(10)<<dis1);

        double mid_t = 0.5*(t0+t1);
        double sdf_mid_t = getDis(p,mid_t+1e-8,traj);

        //中间t的梯度
        double d_mid_t, d_mid_t1;
        //数值法
        // d_mid_t = (sdf_mid_t - getDis(p,mid_t-1e-8,traj))*1e8;
        //解析法 checked
        d_mid_t = getGradT(p,mid_t,traj);

        // INFO_MSG("数值梯度:"<< std::setprecision(10) <<d_mid_t<<",解析梯度:"<<d_mid_t1);

        if(d_mid_t > 0){
          if(sdf_mid_t>dis1){//梯度不会错了 除非是凹的
            INFO_MSG_RED("getSDF convex (or gd) error!!!!!"<<"t0:"<<t0<<",t1:"<<t1<<",err:"<<fabs(t0-t1));
            break;
          }
          t1 = mid_t;
        }
        else{
          if(sdf_mid_t>dis0){
            INFO_MSG_RED("getSDF convex (or gd) error!!!!!!!!!!!!"<<"t0:"<<t0<<",t1:"<<t1<<",err:"<<fabs(t0-t1));
            break;
          }
          t0 = mid_t;
        }

        if(std::fabs(t1-t0)<err){
          end_search = true;
          // INFO_MSG_BLUE("iter:"<< cnt << std::setprecision(10) << ",t_star:" << t_star << ",sdf:" << sdf);
        }
        cnt++;
      }
      t_star = t0; 
      sdf = getDis(p,t0,traj);
      #endif

      // double grad;
      // sdf = smoothedL1(sdf-sdf_min_,grad)+sdf_min_;

      return std::make_pair(sdf,t_star);
    }

  Eigen::Vector3d triLinearInter(const Eigen::Vector3d& diff, double values[2][2][2], const double& resolution){

    double v00 = (1 - diff[0]) * values[0][0][0] + diff[0] * values[1][0][0];
    double v01 = (1 - diff[0]) * values[0][0][1] + diff[0] * values[1][0][1];
    double v10 = (1 - diff[0]) * values[0][1][0] + diff[0] * values[1][1][0];
    double v11 = (1 - diff[0]) * values[0][1][1] + diff[0] * values[1][1][1];
    double v0 = (1 - diff[1]) * v00 + diff[1] * v10;
    double v1 = (1 - diff[1]) * v01 + diff[1] * v11;
    // double dist = (1 - diff[2]) * v0 + diff[2] * v1;

    Eigen::Vector3d grad;
    grad[2] = (v1 - v0) / resolution;
    grad[1] = ((1 - diff[2]) * (v10 - v00) + diff[2] * (v11 - v01)) / resolution;
    grad[0] = (1 - diff[2]) * (1 - diff[1]) * (values[1][0][0] - values[0][0][0]);
    grad[0] += (1 - diff[2]) * diff[1] * (values[1][1][0] - values[0][1][0]);
    grad[0] += diff[2] * (1 - diff[1]) * (values[1][0][1] - values[0][0][1]);
    grad[0] += diff[2] * diff[1] * (values[1][1][1] - values[0][1][1]);
    grad[0] /= resolution;

    // return grad.normalized();
    return grad;
  }

  Eigen::Vector3d getSDFGradBody(const Eigen::Vector3d& pt){

    double resolution = 1e-2;

    Eigen::Vector3d diff;
    diff << 0.5, 0.5, 0.5;

    double values[2][2][2];
    for (int x = 0; x < 2; x++) {
      for (int y = 0; y < 2; y++) {
        for (int z = 0; z < 2; z++) {
            Eigen::Vector3d p;
            p = 2.0*Eigen::Vector3d(x,y,z) - Eigen::Vector3d::Ones();
            p = pt + resolution * p;
            values[x][y][z] = getSDF_body(p);
    }}}

    return triLinearInter(diff, values, resolution).normalized();
  }

  //TODO
  Eigen::Vector3d get2ordSDFGradBody(const Eigen::Vector3d& pt){

    double resolution = 1e-2;

    Eigen::Vector3d diff;
    diff << 0.5, 0.5, 0.5;

    double values[2][2][2];
    for (int x = 0; x < 2; x++) {
      for (int y = 0; y < 2; y++) {
        for (int z = 0; z < 2; z++) {
            Eigen::Vector3d p;
            p = 2.0*Eigen::Vector3d(x,y,z) - Eigen::Vector3d::Ones();
            p = pt + resolution * p;
            values[x][y][z] = getSDF_body(p);
    }}}

    return triLinearInter(diff, values, resolution).normalized();
  }

  std::pair<double,Eigen::Vector3d> getSDFAndGrad(const Eigen::Vector3d& p, 
                                                  const Trajectory<TRAJ_ORDER>& traj){

    double sdf = 9999;
    double t_star;
    Eigen::Vector3d gd;

    sdf = getSDFAndGrad(p, traj, gd, t_star);

    return std::make_pair(sdf,gd);
  }

    //!!!!!!!
  double getSDFAndGrad(const Eigen::Vector3d& p, 
                      const Trajectory<TRAJ_ORDER>& traj,
                      Eigen::Vector3d& gd,
                      double& t_star){
    // double duration = traj.getTotalDuration();
    double sdf = 9999;
    // double t_star;

    // get sdf and t_star
    std::pair<double,double> sdf_t_star = getSDF(p,traj);
    sdf = sdf_t_star.first;
    t_star = sdf_t_star.second;

    Eigen::Vector3d pos, vel, acc, jer;
    Eigen::Vector4d quat;
    Eigen::Vector3d bodyrate;
    cal_uav_state(t_star, traj, pos, vel, acc, jer, quat, bodyrate);

    Eigen::Matrix3d R_b2w = Eigen::Quaterniond(quat[0], quat[1], quat[2], quat[3]).toRotationMatrix();
    Eigen::Vector3d x_rel = R_b2w.transpose()*(p-pos);

    gd = R_b2w*getSDFGradBody(x_rel);

    return sdf;
  }

  double getSDFAndGrad(const Eigen::Vector3d& p, 
                      const Trajectory<TRAJ_ORDER>& traj,
                      const double& t_star,
                      Eigen::Vector3d& gd){
    // double duration = traj.getTotalDuration();
    double sdf = 9999;
    // double t_star;

    // get sdf and t_star
    sdf = getDis(p,t_star,traj);

    Eigen::Vector3d pos, vel, acc, jer;
    Eigen::Vector4d quat;
    Eigen::Vector3d bodyrate;
    cal_uav_state(t_star, traj, pos, vel, acc, jer, quat, bodyrate);

    Eigen::Matrix3d R_b2w = Eigen::Quaterniond(quat[0], quat[1], quat[2], quat[3]).toRotationMatrix();
    Eigen::Vector3d x_rel = R_b2w.transpose()*(p-pos);

    gd = R_b2w*getSDFGradBody(x_rel);

    return sdf;
  }

  //之后尽量用后一个
  Eigen::VectorXd calTrajGrad(const Eigen::Vector3d& p,//[CTG]
                              const Eigen::VectorXd& traj_info){

    int traj_input_size = traj_info.rows()-6;

    // INFO_MSG("----------calTrajGrad-------------");
    Eigen::VectorXd traj_grad(traj_input_size);
    traj_grad.setZero();

    Trajectory<TRAJ_ORDER> traj;
    // Eigen::MatrixXd traj_info_new;
    Eigen::VectorXd traj_info_new(traj_info);

    int n_pie = traj_info.rows()-3;
    n_pie /= 4;

    Eigen::Matrix3d finState(Eigen::Matrix3d::Zero());
    finState.col(0)=Eigen::Vector3d(0.5,0,0);
    Eigen::Map<Eigen::MatrixXd> ini_v_a(traj_info_new.data(),3,2);
    Eigen::Map<Eigen::VectorXd> T(traj_info_new.data()+6,n_pie);
    Eigen::Map<Eigen::MatrixXd> P(traj_info_new.data()+6+n_pie,3,n_pie-1);

    bool generate_new_traj_success;

    // if(debug_mode_){
    //   INFO_MSG("p:"<<p.transpose());
    //   // INFO_MSG("traj_info\n"<<traj_info.transpose());
    //   INFO_MSG("ini_v_a\n"<<ini_v_a);
    //   INFO_MSG("T:"<<T.transpose());
    //   INFO_MSG("P:\n"<<P);

    //   generate_new_traj_success = trajOptPtr_->generate_traj(ini_v_a,finState,T,P,traj);
    //   visPtr_->visualize_traj(traj, "traj0_debug", Eigen::Vector3d(-2,0,0));

    //   std::vector<Eigen::Vector3d> p_vec;
    //   p_vec.push_back(p);
    //   visPtr_->visualize_pointcloud(p_vec, "x_eval0");
    // }

    double epsilon = 1e-6;
    for(int i=0;i<traj_input_size;++i){

        traj_info_new = traj_info;
        traj_info_new(i+6)+=epsilon;

        generate_new_traj_success = generate_traj(ini_v_a,finState,T,P,traj);
        auto sdf_t_new1 = getSDF(p,traj);
        // double sdf_new1 = getSDF(p,traj).first;

        // INFO_MSG("[CTG]1");
        // if(debug_mode_)
        //   visPtr_->visualize_traj(traj, "traj1_debug", Eigen::Vector3d(0,0,0));
        // visSDF(Eigen::Vector3d(0,0,0),traj);

        traj_info_new = traj_info;
        traj_info_new(i+6)-=epsilon;
        generate_new_traj_success = generate_traj(ini_v_a,finState,T,P,traj);
        // double sdf_new0 = getSDF(p,traj).first;
        auto sdf_t_new2 = getSDF(p,traj);
        double gd =  0.5*(sdf_t_new1.first-sdf_t_new2.first)/epsilon;


        traj_grad(i) = gd;

        // if(debug_mode_){
        //   visPtr_->visualize_traj(traj, "traj2_debug", Eigen::Vector3d(2,0,0));
          // visSDF(Eigen::Vector3d(2,0,0),traj);

          // INFO_MSG("d0:" << sdf_t_new1.first <<",t0:" << sdf_t_new1.second
          //     << ",d1:" << sdf_t_new2.first <<",t1:" << sdf_t_new2.second
          //     << ",gd:"<< gd << ",eps:" << epsilon);

        //   if(fabs(gd)>1.5){
        //     INFO_MSG("cin");
        //     int aaa;
        //     std::cin >> aaa;
        // }
        // }

    }

    // if(debug_mode_){
    // INFO_MSG("traj_grad:" << traj_grad.transpose());
    // INFO_MSG("----------calTrajGrad-------------");
    return traj_grad;
  }
  
 
 Eigen::VectorXd calTrajGrad(const Eigen::Vector3d& p,//[CTG]
                             const Eigen::VectorXd& traj_info,
                             const Eigen::MatrixXd& ini_v_a){

    int traj_input_size = traj_info.rows();

    // INFO_MSG("----------calTrajGrad-------------");
    Eigen::VectorXd traj_grad(traj_input_size);
    traj_grad.setZero();

    Trajectory<TRAJ_ORDER> traj;
    // Eigen::MatrixXd traj_info_new;
    Eigen::VectorXd traj_info_new(traj_info);

    int n_pie = traj_info.rows()+3;
    n_pie /= 4;

    Eigen::Matrix3d finState(Eigen::Matrix3d::Zero());
    finState.col(0)=Eigen::Vector3d(0.5,0,0);
    Eigen::Map<Eigen::VectorXd> T(traj_info_new.data(),n_pie);
    Eigen::Map<Eigen::MatrixXd> P(traj_info_new.data()+n_pie,3,n_pie-1);

    bool generate_new_traj_success;
    double epsilon = 1e-6;
    for(int i=0;i<traj_input_size;++i){

      // #ifdef NUM_TRAJ_GD_CAL
      traj_info_new = traj_info;
      traj_info_new(i)+=epsilon;

      generate_new_traj_success = generate_traj(ini_v_a,finState,T,P,traj);
      auto sdf_t_new1 = getSDF(p,traj);

      traj_info_new = traj_info;
      traj_info_new(i)-=epsilon;
      generate_new_traj_success = generate_traj(ini_v_a,finState,T,P,traj);
      // double sdf_new0 = getSDF(p,traj).first;
      auto sdf_t_new2 = getSDF(p,traj);
      double gd =  0.5*(sdf_t_new1.first-sdf_t_new2.first)/epsilon;

      traj_grad(i) = gd;
      // #endif

    }

    // if(debug_mode_){
    // INFO_MSG("traj_grad:" << traj_grad.transpose());
    // INFO_MSG("----------calTrajGrad-------------");
    return traj_grad;
  }

  Eigen::VectorXd calTrajGradNet(const Eigen::Vector3d& p,//[CTGN]
                             const Trajectory<TRAJ_ORDER>& traj){
    // INFO_MSG("[CTGN]"<<"1")
    Eigen::VectorXd traj_gd=calTrajGrad(p,traj);
    int n_pie = traj.getPieceNum();

    Eigen::Map<Eigen::VectorXd> gdT(traj_gd.data(),n_pie);
    Eigen::Map<Eigen::MatrixXd> gdP(traj_gd.data()+n_pie,3,n_pie-1);

    gdT/=traj.getScale();

    for(int i=0;i<n_pie-1;++i)
      gdP.col(i)=ptWorld2Net_Rot(gdP.col(0));

    return traj_gd;
  }



 Eigen::VectorXd calTrajGrad(const Eigen::Vector3d& p,//[CTG]
                             const Trajectory<TRAJ_ORDER>& traj,
                             const double epsilon = 1e-5){

    // INFO_MSG("calTrajGrad 1");

    int n_pie = traj.getPieceNum();
    int traj_input_size = 4*n_pie-3;

    // INFO_MSG("----------calTrajGrad-------------");
    Eigen::VectorXd traj_grad(traj_input_size);
    traj_grad.setZero();



    Eigen::Matrix3d iniState, finState;

    traj.getInOutStates(iniState,finState);

    Eigen::MatrixXd mid_p=traj.getPositions().block(0,1,3,n_pie-1);
    Eigen::Map<Eigen::VectorXd> P(mid_p.data(),3*(n_pie-1));
    Eigen::VectorXd T=traj.getDurations();
    Eigen::VectorXd traj_info(traj_input_size);
    traj_info << T,P;

    // INFO_MSG("calTrajGrad 2");

    Trajectory<TRAJ_ORDER> traj_new;
    Eigen::VectorXd traj_info_new(traj_info);
    Eigen::Map<Eigen::VectorXd> T_new(traj_info_new.data(),n_pie);
    Eigen::Map<Eigen::MatrixXd> P_new(traj_info_new.data()+n_pie,3,n_pie-1);

    //数值方法初始t从相同位置开始
    auto sdf_t = getSDF(p,traj);
    // INFO_MSG("t:"<<sdf_t.second);

    bool generate_new_traj_success;
    // double epsilon = 1e-5;
    for(int i=0;i<traj_input_size;++i){

      traj_info_new = traj_info;
      traj_info_new(i)+=epsilon;

      generate_new_traj_success = generate_traj2(iniState,finState,T_new,P_new,traj_new);
      auto sdf_t_new1 = getSDF(p,traj_new,sdf_t.second);

      traj_info_new = traj_info;
      traj_info_new(i)-=epsilon;
      generate_new_traj_success = generate_traj2(iniState,finState,T_new,P_new,traj_new);
      // double sdf_new0 = getSDF(p,traj).first;
      auto sdf_t_new2 = getSDF(p,traj_new,sdf_t.second);
      double gd =  0.5*(sdf_t_new1.first-sdf_t_new2.first)/epsilon;

      traj_grad(i) = gd;
    }

    return traj_grad;
  }
  
  void calSDFAndGd(const Eigen::VectorXd& traj_info,
                  const Eigen::MatrixXd& ini_v_a,
                  const Eigen::MatrixXd& pts,
                  Eigen::VectorXd& sdf,
                  Eigen::MatrixXd& gd){
    Eigen::Vector3d pt;
    for(int i=0;i<pts.cols();++i){
      pt=pts.col(i);
      sdf(i)=getSDF(pt,trajInfo2traj(traj_info,ini_v_a)).first;
      gd.col(i)=calTrajGrad(pt,traj_info,ini_v_a);
    }
  }

// With Nesterov Momentum Acceleration
//momentum=0即不使用Nesterov Momentum Acce
void gradient_descent(double momentum,double t_min,double t_max,
                      const std::function<double(const double)>& f,
                      const std::function<double(const double)>& gf,
                      const double x0, double &fx, double &x)
{
    assert((t_max>0)&&(t_max<100)&&" in gradient descent,t_max must > 0 and t_max <100");
    assert((t_min>=0)&&" in gradient descent,t_min must >=0");
    assert((momentum>=0)&&(momentum<=1)&&"momentum must between 0~1");
// int max_iter = 300;
    int max_iter = 1000;
    // double alpha = 0.02;
    //初始步长
    double alpha = 5e-3; // CHANGE THIS BACK,alpha是迭代的初始数值back tracking linesearch
    double tau = alpha;
    double g   = 0.0;//tao是back tracking linesearch的补长
    // double min = t_min;
    // double max = t_max;
    double tol = 1e-10;//x error
    double xmin = x0; // x0时间初始数值.fx梯度下降之后的函数数值,x优化后的时间
    double xmax = x0;
    x = x0;

    double projection=0;
    double change=0;
    
    double prev_x = 10000000.0;
    int iter = 0;
    bool stop = false;
    double x_candidate, fx_candidate;
    g = 100.0;
    // std::cout << "run" << std::endl;
    // int in_existing_interval = -1;
    assert(iter < max_iter && !stop && abs(x - prev_x) > tol);//换成自己的pos_eval后评估导致assert iter<max_iter
    //原先有这个assert不知道为什么替换成自己的pos_eval后出shit!??????????????????
    //原先有这个assert不知道为什么替换成自己的pos_eval后出shit!??????????????????
    //原先有这个assert不知道为什么替换成自己的pos_eval后出shit!??????????????????
    // #define GNOW ros::Time::now()
    // #define GMS(x,y) (x-y).toSec()*1000
    // ros::Time t1,t2;
    int div_inner = 10;

    // INFO_MSG("x:"<<std::setprecision(10)<<x<<",fx:"<<fx);
    while (iter < max_iter && !stop && abs(x - prev_x) > tol)//迭代次数没有满足或者精度残差没有收敛
    {
        xmin = std::min(xmin, x);
        xmax = std::max(xmax, x);


        fx = f(x);
        g = gf(x);

        // 不超过搜索方向极限步长
        tau = g>0?x-xmin:xmax-x;
        tau = std::min(tau, alpha);

        tau = alpha;
        prev_x = x;

        // t1 = GNOW;
        if(abs(fx)> 1e5){std::cout<<"fx = "<<fx <<",g:"<<g<<std::endl;}
        for (int div = 1; div < div_inner; div++)//里面一轮最多进行div_inner次迭代
        {
            iter = iter + 1;
            assert(iter < max_iter);
            //思考一下是否需要momentum的正负号问题？
            projection=x+momentum*change;
            g=gf(projection);
            change=momentum*change-tau * ((double)(g > 0) - (g < 0));
             //原先有这个assert不知道为什么替换成自己的pos_eval后出shit!??????????????????
             //原先有这个assert不知道为什么替换成自己的pos_eval后出shit!??????????????????
             //原先有这个assert不知道为什么替换成自己的pos_eval后出shit!??????????????????
            // if(iter==max_iter)
            // {
            //     break;
            // }
            // x_candidate = x - tau * ((double)(g > 0) - (g < 0));
            x_candidate=x+change;
            x_candidate = std::max(std::min(x_candidate, t_max), t_min);//时间限制幅度在0~1有shit!!!!!!!!!!!!!时间不能只限制在0~1应该是轨迹时间t_max与t_min
            fx_candidate = f(x_candidate);
            if(abs(fx_candidate)> 1e5){std::cout<<"fx_candidate = "<<fx_candidate <<std::endl;}
            
            // 必须在一条直线下方
            // l(α) = φ(0) + c_1*α*的d_f(xk)*d_k
            // t_seed已经很小,此处的梯度几乎在0.01这个这个量级
            // alpha 在 0.5 为啥很难达到

            double k = 0.45;
            if ((fx_candidate - fx) < (k * (x_candidate - x) * g))//汪博L1优化-63页Armijo line search满足,退出for循环
            {//line search满足,更新一次x,fx
                x = x_candidate;
                // std::cout<<"xx = "<<x <<" change = "<<change <<std::endl;
                fx = fx_candidate;

                // std::cout << div << std::endl;
                break;
            }
            tau = 0.5 * tau;
            if (div == div_inner-1)
            {
                // std::cout << div << std::endl;
                stop = true;//内层9次都没有满足line search条件?放弃迭代
            }

            // double d_fx = (fx_candidate - fx)/(x_candidate - x);
            // INFO_MSG("x_candidate:"<<std::setprecision(10)<<x_candidate
            //           <<",fx_candidate:"<<fx_candidate
            //           <<",d_fx:"<<d_fx<<",d_fx/g:"<<d_fx/g);
        }
        // t2 = GNOW;
        // std::cout<<"4:"<<GMS(t2,t1)<<" ms"<<std::endl;
        // 耗时0.002ms

        // INFO_MSG("x-prex="<<std::setprecision(10)<<x-prev_x<<",t:"<<x<<",SDF:"<<fx);
    }

    // INFO_MSG_GREEN("t_star:"<<std::setprecision(10)<<x<<",SDF:"<<fx);

}


Eigen::Matrix3d getQuatTransDW(const Eigen::Vector4d &quat)
{ // 旋转矩阵R转成四元数quat，返回R.transpose() 的四元数系数W的求导
    Eigen::Matrix3d ret;
    double w = quat(0);
    double x = quat(1);
    double y = quat(2);
    double z = quat(3);
    ret << 0, 2 * z, -2 * y,
        -2 * z, 0, 2 * x,
        2 * y, -2 * x, 0;
    return ret;
}
Eigen::Matrix3d getQuatTransDX(const Eigen::Vector4d &quat)
{ // 旋转矩阵R转成四元数quat，返回R.transpose() 的四元数系数W的求导
    Eigen::Matrix3d ret;
    double w = quat(0);
    double x = quat(1);
    double y = quat(2);
    double z = quat(3);
    ret << 0, 2 * y, 2 * z,
        2 * y, -4 * x, 2 * w,
        2 * z, -2 * w, -4 * x;
    return ret;
}
Eigen::Matrix3d getQuatTransDY(const Eigen::Vector4d &quat)
{ // 旋转矩阵R转成四元数quat，返回R.transpose() 的四元数系数W的求导
    Eigen::Matrix3d ret;
    double w = quat(0);
    double x = quat(1);
    double y = quat(2);
    double z = quat(3);
    ret << -4 * y, 2 * x, -2 * w,
        2 * x, 0, 2 * z,
        2 * w, 2 * z, -4 * y;
    return ret;
}
Eigen::Matrix3d getQuatTransDZ(const Eigen::Vector4d &quat)
{ // 旋转矩阵R转成四元数quat，返回R.transpose() 的四元数系数W的求导
    Eigen::Matrix3d ret;
    double w = quat(0);
    double x = quat(1);
    double y = quat(2);
    double z = quat(3);
    ret << -4 * z, 2 * w, 2 * x,
        -2 * w, -4 * z, 2 * y,
        2 * x, 2 * y, 0;
    return ret;
}



};

}

#endif
