#pragma once
#include <decomp_ros_utils/data_ros_utils.h>
#include <decomp_util/ellipsoid_decomp.h>
#include <mapping/mapping.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>

#include <Eigen/Core>
#include <memory>
#include <queue>
#include <traj_opt/geoutils.hpp>
#include <unordered_map>
#include <rotation_util/rotation_util.hpp>
#include <visualization/visualization.hpp>
#include <env/env.hpp>

namespace env {
// modeled as an rectancle 
class TargetBBX{
private:
  Eigen::Vector3d center_;
  Eigen::Vector3d vel_;
  Eigen::Quaterniond q_;
  std::vector<Eigen::Vector3d> vps_origin_; // 8 origin vertexes
  std::vector<Eigen::Vector3d> vps_;  // 8 vertexes

  std::vector<Eigen::Vector3d> n_origin_;  // 6 normal of plane
  std::vector<Eigen::Vector3d> n_;  // 6 normal of plane
  std::vector<Eigen::Vector3d> p_;  // 6 point on plane
public: 
  typedef std::shared_ptr<TargetBBX> Ptr;
  // init convex polytope info: the normal of the plane & the points on the plane
  void setBBX(const double& x_len, const double& y_len, const double& z_len){
    vps_origin_.clear();
    double x_add_len = 5.0;
    vps_origin_.emplace_back(-x_len/2.0, -y_len/2.0, -z_len/2.0);
    vps_origin_.emplace_back(-x_len/2.0, +y_len/2.0, -z_len/2.0);
    vps_origin_.emplace_back(x_add_len+x_len/2.0, +y_len/2.0, -z_len/2.0);
    vps_origin_.emplace_back(x_add_len+x_len/2.0, -y_len/2.0, -z_len/2.0);
    vps_origin_.emplace_back(-x_len/2.0, -y_len/2.0, +z_len/2.0);
    vps_origin_.emplace_back(-x_len/2.0, +y_len/2.0, +z_len/2.0);
    vps_origin_.emplace_back(x_add_len+x_len/2.0, +y_len/2.0, +z_len/2.0);
    vps_origin_.emplace_back(x_add_len+x_len/2.0, -y_len/2.0, +z_len/2.0);
    //TODO object specific offset
    for(auto& p:vps_origin_){
      p(2) = p(2) - z_len/2.0 - 0.1;
    }
    vps_ = vps_origin_;

    n_origin_.emplace_back(1.0, 0.0, 0.0);
    n_origin_.emplace_back(-1.0, 0.0, 0.0);
    n_origin_.emplace_back(0.0, 1.0, 0.0);
    n_origin_.emplace_back(0.0, -1.0, 0.0);
    n_origin_.emplace_back(0.0, 0.0, 1.0);
    n_origin_.emplace_back(0.0, 0.0, -1.0);
    n_ = n_origin_;
    p_.push_back(vps_[2]);
    p_.push_back(vps_[0]);
    p_.push_back(vps_[1]);
    p_.push_back(vps_[3]);
    p_.push_back(vps_[5]);
    p_.push_back(vps_[0]);
  }
  // Update BBX pos and orientation
  void updateBBX(const Eigen::Vector3d& center,const Eigen::Vector3d& vel, const Eigen::Quaterniond q){
    center_ = center;
    vel_ = vel;
    q_ = q;
    // Update vertexes
    for(size_t i = 0; i < vps_.size(); i++){
      vps_[i] = q_.toRotationMatrix()*vps_origin_[i] + center_;
    }
    // Update normal
    for(size_t i = 0; i < p_.size(); i++){
      n_[i] = q_.toRotationMatrix()*n_origin_[i];
    }
    // Update point on the plane
    p_[0] = vps_[2];
    p_[1] = vps_[0];
    p_[2] = vps_[1];
    p_[3] = vps_[3];
    p_[4] = vps_[5];
    p_[5] = vps_[0];
  }
  // Return if the query point in the BBX
  bool isInBBX(Eigen::Vector3d& qp) const{
    for(size_t i = 0; i < p_.size(); i++){
      if(n_[i].dot(qp-p_[i]) >= -1e-4){ // For numerical stabibity
        return false;
      }
    }
    return true;
  }
  std::vector<Eigen::Vector3d> getVps() const{
    return vps_;
  }
  Eigen::Vector3d getCenter() const{
    return center_;
  }
  Eigen::Vector3d getVel() const{
    return vel_;
  }
  Eigen::VectorXd getHyperPlane(const int i) const{
    if(i >= 6){
      std::cout << "Target BBX getHyperPlane error!" <<std::endl;
    }
    Eigen::VectorXd hp;
    hp.resize(6);
    hp << n_[i], p_[i];
    return hp;
  }
};

class EnvSfc {
  static constexpr int MAX_MEMORY = 1 << 18;
  static constexpr double MAX_DURATION = 0.2;

 private:
  ros::Publisher hPolyPub_;
  std::shared_ptr<mapping::OccGridMap> mapPtr_;
  std::shared_ptr<visualization::Visualization> visPtr_;
  TargetBBX::Ptr target_bbx_ptr_;

  Eigen::Vector3d odom_p_, odom_v_;
  bool isdebug_ = false;

 public:
  EnvSfc(ros::NodeHandle& nh,
      std::shared_ptr<mapping::OccGridMap>& mapPtr) : mapPtr_(mapPtr) {
    hPolyPub_ = nh.advertise<decomp_ros_msgs::PolyhedronArray>("polyhedra", 1);
  }

  inline void set_target_ptr(TargetBBX::Ptr target_bbx_ptr){
    target_bbx_ptr_ = target_bbx_ptr;
  }

  void compressPoly(Polyhedron3D& poly, double dx) {
    vec_E<Hyperplane3D> hyper_planes = poly.hyperplanes();
    for (uint j = 0; j < hyper_planes.size(); j++) {
      hyper_planes[j].p_ = hyper_planes[j].p_ - hyper_planes[j].n_ * dx;
    }
    poly = Polyhedron3D(hyper_planes);
  }
  void compressPoly(Eigen::MatrixXd& poly, double dx) {
    for (int i = 0; i < poly.cols(); ++i) {
      poly.col(i).tail(3) = poly.col(i).tail(3) - poly.col(i).head(3) * dx;
    }
  }

  void getPointCloudAroundLine(const vec_Vec3f& line,
                               const int maxWidth,
                               vec_Vec3f& pc) {
    pc.clear();
    Eigen::Vector3d p0 = line.front();
    Eigen::Vector3d p1 = line.back();
    Eigen::Vector3i idx0 = mapPtr_->pos2idx(p0);
    Eigen::Vector3i idx1 = mapPtr_->pos2idx(p1);
    Eigen::Vector3i d_idx = idx1 - idx0;
    Eigen::Vector3i step = d_idx.array().sign().cast<int>();
    Eigen::Vector3d delta_t;
    Eigen::Vector3i tmp_p, margin;
    margin.setConstant(maxWidth);
    for (tmp_p.x() = idx0.x() - margin.x(); tmp_p.x() <= idx0.x() + margin.x(); ++tmp_p.x()) {
      for (tmp_p.y() = idx0.y() - margin.y(); tmp_p.y() <= idx0.y() + margin.y(); ++tmp_p.y()) {
        for (tmp_p.z() = idx0.z() - margin.z(); tmp_p.z() <= idx0.z() + margin.z(); ++tmp_p.z()) {
          Eigen::Vector3d tmp_pt = mapPtr_->idx2pos(tmp_p);
          if (mapPtr_->isOccupied(tmp_p)) { //|| target_bbx_ptr_->isInBBX(tmp_pt)
            pc.push_back(mapPtr_->idx2pos(tmp_p));
          }
        }
      }
    }
    for (int i = 0; i < 3; ++i) {
      delta_t(i) = d_idx(i) == 0 ? 2.0 : 1.0 / std::abs(d_idx(i));
    }
    Eigen::Vector3d t_max;
    for (int i = 0; i < 3; ++i) {
      t_max(i) = step(i) > 0 ? std::ceil(p0(i)) - p0(i) : p0(i) - std::floor(p0(i));
    }
    t_max = t_max.cwiseProduct(delta_t);
    Eigen::Vector3i rayIdx = idx0;
    // ray casting
    while (rayIdx != idx1) {
      // find the shortest t_max
      int s_dim = 0;
      for (int i = 1; i < 3; ++i) {
        s_dim = t_max(i) < t_max(s_dim) ? i : s_dim;
      }
      rayIdx(s_dim) += step(s_dim);
      t_max(s_dim) += delta_t(s_dim);
      margin.setConstant(maxWidth);
      margin(s_dim) = 0;
      Eigen::Vector3i center = rayIdx;
      center(s_dim) += maxWidth * step(s_dim);
      for (tmp_p.x() = center.x() - margin.x(); tmp_p.x() <= center.x() + margin.x(); ++tmp_p.x()) {
        for (tmp_p.y() = center.y() - margin.y(); tmp_p.y() <= center.y() + margin.y(); ++tmp_p.y()) {
          for (tmp_p.z() = center.z() - margin.z(); tmp_p.z() <= center.z() + margin.z(); ++tmp_p.z()) {
            if (mapPtr_->isOccupied(tmp_p)) {
              pc.push_back(mapPtr_->idx2pos(tmp_p));
            }
          }
        }
      }
    }
  }


  void generateOneCorridor(const std::pair<Eigen::Vector3d, Eigen::Vector3d>& l,
                           const double bbox_width,
                           Eigen::MatrixXd& hPoly) {
    vec_Vec3f obs_pc;
    EllipsoidDecomp3D decomp_util;
    decomp_util.set_local_bbox(Eigen::Vector3d(bbox_width, bbox_width, bbox_width));
    int maxWidth = bbox_width / mapPtr_->resolution;

    vec_Vec3f line;
    line.push_back(l.first);
    line.push_back(l.second);
    getPointCloudAroundLine(line, maxWidth, obs_pc);
    decomp_util.set_obs(obs_pc);
    decomp_util.dilate(line);
    Polyhedron3D poly = decomp_util.get_polyhedrons()[0];
    compressPoly(poly, 0.1);

    vec_E<Hyperplane3D> current_hyperplanes = poly.hyperplanes();
    hPoly.resize(6, current_hyperplanes.size());
    for (uint j = 0; j < current_hyperplanes.size(); j++) {
      hPoly.col(j) << current_hyperplanes[j].n_, current_hyperplanes[j].p_;
    }
    return;
  }

  void hullSplit(Polyhedron3D& poly, vec_E<Polyhedron3D>& ret_polys){
    Eigen::MatrixXd current_poly;
    vec_E<Hyperplane3D> current_hyperplanes = poly.hyperplanes();
    current_poly.resize(6, current_hyperplanes.size());
    for (uint j = 0; j < current_hyperplanes.size(); j++) {
      current_poly.col(j) << current_hyperplanes[j].n_, current_hyperplanes[j].p_;
      //outside
    }
    
    Eigen::MatrixXd vPoly;
    geoutils::enumerateVs(current_poly, vPoly); // 对于当前这个凸包操作，vPoly是这个凸包的所有顶点
    std::vector<Eigen::Matrix<double, 6, -1>>checked_polys;
    std::vector<Eigen::Vector3d> gmids;

        bool deledge = false; 

        for(int m = 0 ; m < 6 ; ++m)//全部递归时间1-2ms  对于obb BBX的6个面来说
        {
          for(int j = 0 ; j < vPoly.cols() ; ++j) // 第j个凸包顶点
          {   
            double cross_m = (vPoly.col(j).transpose() - target_bbx_ptr_->getHyperPlane(m).head(3).transpose()) * target_bbx_ptr_->getHyperPlane(m).tail(3);
            if(cross_m > 0)  // 如果存在凸包顶点在面的外面，该面可以去分割poly
            {
                deledge = true;
                const Eigen::Matrix<double, 6, -1> ConPoly = current_poly;
                Eigen::Matrix<double, 6, -1> tempPoly = current_poly;
                int num = tempPoly.cols(); // 当前凸包所含超平面个数
                tempPoly.resize(6,num+1); // 预留出新增的BBX超平面位置

                tempPoly.leftCols(num) = ConPoly;
                tempPoly.col(num).head<3>() = -1* target_bbx_ptr_->getHyperPlane(m).head(3);
                tempPoly.col(num).tail<3>() = target_bbx_ptr_->getHyperPlane(m).tail(3);

                Eigen::MatrixXd v_temp; // 分割后凸包的顶点
                if(!geoutils::enumerateVs(tempPoly, v_temp))  // 没有相交区域，未相交
                {
                    break;
                }
                
                int che_num = 0;//是否与已分裂出来的凸包属于包含关系(这么判断其实不准确)  几何中心 + 顶点数判断
                Eigen::MatrixXd v_combP;
                Eigen::MatrixXd v_old;
                Eigen::Vector3d comb_poly_mid;
                Eigen::Vector3d checked_polys_mid;
                Eigen::Vector3d tempPoly_mid;
                tempPoly_mid = v_temp.rowwise().sum() / v_temp.cols();
                for(size_t n = 0 ; n < checked_polys.size() ; ++n)
                {
                    Eigen::Matrix<double,6,-1> comb_poly ; // 生成checked_poly和当前凸包的交集凸包
                    int cols1 = checked_polys[n].cols();
                    int cols2 = tempPoly.cols();
                    comb_poly.resize(6,cols1+cols2);
                    comb_poly.leftCols(cols1) = checked_polys[n];
                    comb_poly.rightCols(cols2) = tempPoly;

                    if(!geoutils::enumerateVs(comb_poly, v_combP)) // 交集为空，无重合部分，不包含
                    {
                        che_num++;
                        continue; 
                    }
                    geoutils::enumerateVs(checked_polys[n], v_old);
                    comb_poly_mid = v_combP.rowwise().sum() / v_combP.cols();
                    checked_polys_mid = v_old.rowwise().sum() / v_old.cols();
                    
                    if(((comb_poly_mid - checked_polys_mid).norm() < 1e-3) &&(v_combP.cols() == v_old.cols())) //旧属于新，和交集顶点数相等的就是小的集合
                    {
                        checked_polys[n] = tempPoly;
                        gmids[n] = tempPoly_mid;
                        break;
                    }                         
                    if(((comb_poly_mid - tempPoly_mid).norm() < 1e-3)&&(v_combP.cols() == v_temp.cols())) //新属于旧  
                    {
                        break;
                    }
                    che_num++;
                }
                  if(che_num == checked_polys.size()) // 未找到包含关系
                {
                    checked_polys.push_back(tempPoly);
                    gmids.push_back(tempPoly_mid);
                }
                break;
            }
          }
        }
        if(checked_polys.size() == 0)  
        {
            deledge = true;
            // poly->flag = 2;
            // poly->cover_obbIdx.push_back(index + 1);
            // continue;
            std::cout << " split fail! " << std::endl;
            return;
        }
        ret_polys.clear();
        for(auto& polys:checked_polys){
          vec_E<Hyperplane3D> hyper_planes;
          hyper_planes.resize(polys.cols());
          for (int i = 0; i < polys.cols(); ++i) {
            hyper_planes[i].n_ = polys.col(i).head(3);
            hyper_planes[i].p_ = polys.col(i).tail(3);
          }
          ret_polys.emplace_back(hyper_planes);
        }
  }

  void handleTargetBBX(const std::vector<Eigen::Vector3d>& path, vec_E<Polyhedron3D>& decompPolys, vec_E<Polyhedron3D>& ret_polys){
    ret_polys.clear();
    vec_E<Polyhedron3D> all_polys;
    for(auto& poly:decompPolys){
      if(poly.inside(target_bbx_ptr_->getCenter())){
        vec_E<Polyhedron3D> splited_polys;
        hullSplit(poly, splited_polys);

        std::vector<bool> ispicked(splited_polys.size(), false);
        for(auto& p:path){
          for(size_t i = 0; i < splited_polys.size(); i++){
            if(!ispicked[i]){
              if(splited_polys[i].inside(p)){
                ispicked[i] = true;
                ret_polys.push_back(splited_polys[i]);
              }
            }
          }
        }
      }else{
        ret_polys.push_back(poly);
      }
    }


    std::cout << "poly gid: "<<std::endl;
    for(auto poly : ret_polys){
      Eigen::Vector3d mid = Eigen::Vector3d::Zero();
      vec_E<Hyperplane3D> current_hyperplanes = poly.hyperplanes();
      for (uint j = 0; j < current_hyperplanes.size(); j++){
        mid += current_hyperplanes[j].p_;
      }
      mid = mid / current_hyperplanes.size();
      std::cout << "son: "<<mid.transpose() << std::endl;
    }
  }

  void polys2MatrixVec(const vec_E<Polyhedron3D> poly_sur, std::vector<Eigen::MatrixXd>& poly_tar){
    poly_tar.clear();
    Eigen::MatrixXd current_poly;
    for (uint i = 0; i < poly_sur.size(); i++) {
      current_poly = polys2Matrix(poly_sur[i]);
      poly_tar.push_back(current_poly);
    }
  }

  Eigen::MatrixXd polys2Matrix(const Polyhedron3D& poly_sur){
    Eigen::MatrixXd poly_tar;
    vec_E<Hyperplane3D> current_hyperplanes = poly_sur.hyperplanes();
    poly_tar.resize(6, current_hyperplanes.size());
    for (uint j = 0; j < current_hyperplanes.size(); j++) {
      poly_tar.col(j) << current_hyperplanes[j].n_, current_hyperplanes[j].p_;
      //outside
    }
    return poly_tar;
  }

  void matrix2PolyVec(const std::vector<Eigen::MatrixXd>& poly_sur, vec_E<Polyhedron3D> poly_tar){
    poly_tar.clear();
    for (uint i = 0; i < poly_sur.size(); i++) {
      Polyhedron3D current_poly;
      matrix2Poly(poly_sur[i], current_poly);
      poly_tar.push_back(current_poly);
    }
  }

  void matrix2Poly(const Eigen::MatrixXd& poly_sur, Polyhedron3D& poly_tar){
    poly_tar.vs_.clear();
    for (int j = 0; j < poly_sur.cols(); j++){
      Hyperplane3D hp(poly_sur.col(j).tail(3), poly_sur.col(j).head(3));
      poly_tar.vs_.push_back(hp);
    }
  }

  Polyhedron3D matrix2Poly(const Eigen::MatrixXd& poly_sur){
    Polyhedron3D poly_tar;
    for (int j = 0; j < poly_sur.cols(); j++){
      Hyperplane3D hp(poly_sur.col(j).tail(3), poly_sur.col(j).head(3));
      poly_tar.vs_.push_back(hp);
    }
    return poly_tar;
  }

  void selectOneStartPoly(std::vector<Eigen::MatrixXd>& hPolys, const std::vector<Eigen::Vector3d>& path){
    // if startpoint contain by one more convex, delete redunt
    DEBUG_MSG("hPolys: " << hPolys.size());
    if(hPolys.size() > 1){
      std::vector<Eigen::MatrixXd> ret_polys;
      std::vector<Eigen::MatrixXd> other_polys;
      ret_polys.clear();
      other_polys.clear();
      std::vector<Polyhedron3D> start_polys;
      for(auto poly:hPolys){
        Polyhedron3D cur_poly = matrix2Poly(poly);
        if(cur_poly.inside(odom_p_)){
          start_polys.push_back(cur_poly);
        }else{
          other_polys.push_back(poly);
        }
      }
      DEBUG_MSG("start_polyssize: " << start_polys.size());

      // delete startpolys which did not intersect with the next one
      if(start_polys.size() > 1){
        if(other_polys.size() > 0){
          DEBUG_MSG("other_polys.size(): " << other_polys.size());
          for(auto it = start_polys.begin(); it != start_polys.end(); ){
            Eigen::MatrixXd start_po = polys2Matrix(*it);
            Eigen::Matrix<double,6,-1> comb_poly ; // 生成checked_poly和当前凸包的交集凸包
            int cols1 = start_po.cols();
            int cols2 = other_polys[0].cols();
            comb_poly.resize(6,cols1+cols2);
            comb_poly.leftCols(cols1) = start_po;
            comb_poly.rightCols(cols2) = other_polys[0];
            Eigen::MatrixXd v_combP;
            if(!geoutils::enumerateVs(comb_poly, v_combP)) // 交集为空，无重合部分，不包含
            {
              it = start_polys.erase(it);
              if(start_polys.size() == 1) break;
            }else{
              it++;
            }
          }
          DEBUG_MSG("other_polys.size() > 0, start_polyssize: " << start_polys.size());
        }else{
          for(auto it = start_polys.begin(); it != start_polys.end(); ){
            if(!it->inside(path.back())) // 交集为空，无重合部分，不包含
            {
              it = start_polys.erase(it);
              if(start_polys.size() == 1) break;
            }else{
              it++;
            }
          }
          DEBUG_MSG("other_polys.size() = 0, start_polyssize: " << start_polys.size());
        }
      }


      if(start_polys.size() > 1){
        // if startpoint contain by one more convex, select one extend more along odomv, and delete others
        if(odom_v_.norm() > 0.5){
          for(Eigen::Vector3d p = odom_p_; ; p += odom_v_.normalized()*0.1){
            for(auto it = start_polys.begin(); it != start_polys.end(); ){
              if(!it->inside(p)){
                it = start_polys.erase(it);
                if(start_polys.size() == 1) break;
              }else{
                it++;
              }
            }
            if(start_polys.size() == 1) break;
          }
        }else{
          // if startpoint contain by one more convex, select one extend more along path, and delete others
          for(auto p:path){
            for(auto it = start_polys.begin(); it != start_polys.end(); ){
              if(!it->inside(p)){
                it = start_polys.erase(it);
                if(start_polys.size() == 1) break;
              }else{
                it++;
              }
            }
            if(start_polys.size() == 1) break;
          }
        }
      }

      ret_polys.push_back(polys2Matrix(start_polys[0]));
      for(auto poly:other_polys){
        ret_polys.push_back(poly);
      }
      hPolys = ret_polys;
    }
  }

  bool filterCorridor(std::vector<Eigen::MatrixXd>& hPolys, const std::vector<Eigen::Vector3d>& path) {
    // return false;
    bool ret = false;
    if (hPolys.size() <= 2) {
      return ret;
    }
    std::vector<Eigen::MatrixXd> ret_polys;
    Eigen::MatrixXd hPoly0 = hPolys[0];
    Eigen::MatrixXd curIH;
    Eigen::Vector3d interior;
    for (int i = 2; i < (int)hPolys.size(); i++) {
      curIH.resize(6, hPoly0.cols() + hPolys[i].cols());
      curIH << hPoly0, hPolys[i];
      if (geoutils::findInteriorDist(curIH, interior) < 1.0) {
        ret_polys.push_back(hPoly0);
        hPoly0 = hPolys[i - 1];
      } else {
        ret = true;
      }
    }
    ret_polys.push_back(hPoly0);
    ret_polys.push_back(hPolys.back());
    hPolys = ret_polys;
    return ret;
  }



  void generateSFC(const std::vector<Eigen::Vector3d>& path,
                   const Eigen::Vector3d& bbox_width,
                   std::vector<Eigen::MatrixXd>& hPolys,
                   std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& keyPts, bool handle_target = false) {
    assert(path.size() > 1);
    vec_Vec3f obs_pc;
    EllipsoidDecomp3D decomp_util;
    decomp_util.set_local_bbox(bbox_width);

    int maxWidth = bbox_width.maxCoeff() / mapPtr_->resolution;

    vec_E<Polyhedron3D> decompPolys;

    int path_len = path.size();

    int idx = 0;
    keyPts.clear();

    while (idx < path_len - 1) {
      int next_idx = idx;
      // looking forward -> get a farest next_idx
      while (next_idx + 1 < path_len && checkRayValid(path[idx], path[next_idx + 1], bbox_width.maxCoeff())) {
        next_idx++;
      }
      // generate corridor with idx and next_idx
      vec_Vec3f line;
      line.push_back(path[idx]);
      line.push_back(path[next_idx]);
      keyPts.emplace_back(path[idx], path[next_idx]);
      getPointCloudAroundLine(line, maxWidth, obs_pc);
      decomp_util.set_obs(obs_pc);
      decomp_util.dilate(line);
      Polyhedron3D poly = decomp_util.get_polyhedrons()[0];
      decompPolys.push_back(poly);

      // find a farest idx in current corridor
      idx = next_idx;
      while (idx + 1 < path_len && decompPolys.back().inside(path[idx + 1])) {
        idx++;
      }
    }

    // Fetch the poly contain the target
    // visCorridor(decompPolys);

    if(handle_target){
      vec_E<Polyhedron3D> ret_polys;
      handleTargetBBX(path, decompPolys, ret_polys);

      if(isdebug_){
        visCorridor(ret_polys);
        int a;
        std::cin >> a;
      }

      polys2MatrixVec(ret_polys, hPolys);
      selectOneStartPoly(hPolys, path);
      filterCorridor(hPolys, path); // elimate redundant poly
    }else{
      polys2MatrixVec(decompPolys, hPolys);
      filterCorridor(hPolys, path);
    }



    if(isdebug_){
      visCorridor(hPolys);
      int a;
      std::cin >> a;
    }

    // check again
    Eigen::MatrixXd curIH;
    Eigen::Vector3d interior;
    Eigen::MatrixXd current_poly = hPolys.back();
    std::vector<int> inflate(hPolys.size(), 0);
    // tranverse all the poly, if its minmax distance  > 0.1, compress it; else label it as inflate
    for (int i = 0; i < (int)hPolys.size(); i++) {
      if (geoutils::findInteriorDist(current_poly, interior) < 0.1) {
        inflate[i] = 1;
      } else {
        compressPoly(hPolys[i], 0.1);
      }
    }
    // if compress results Interior = empty, inflate back
    for (int i = 1; i < (int)hPolys.size(); i++) {
      curIH.resize(6, hPolys[i - 1].cols() + hPolys[i].cols());
      curIH << hPolys[i - 1], hPolys[i];
      if (!geoutils::findInterior(curIH, interior)){
        if (!inflate[i - 1]) {
          compressPoly(hPolys[i - 1], -0.1);
          inflate[i - 1] = 1;
        }
      } else {
        continue;
      }
      curIH << hPolys[i - 1], hPolys[i];
      if (!geoutils::findInterior(curIH, interior)) {
        if (!inflate[i]) {
          compressPoly(hPolys[i], -0.1);
          inflate[i] = 1;
        }
      }
    }


    

  }

  bool inline checkRayValid(const Eigen::Vector3d& p0, const Eigen::Vector3d& p1, double max_dist) const {
    Eigen::Vector3d dp = p1 - p0;
    double dist = dp.norm();
    if (dist > max_dist) {
      return false;
    }
    Eigen::Vector3i idx0 = mapPtr_->pos2idx(p0);
    Eigen::Vector3i idx1 = mapPtr_->pos2idx(p1);
    Eigen::Vector3i d_idx = idx1 - idx0;
    Eigen::Vector3i step = d_idx.array().sign().cast<int>();
    Eigen::Vector3d delta_t;
    for (int i = 0; i < 3; ++i) {
      delta_t(i) = dp(i) == 0 ? std::numeric_limits<double>::max() : 1.0 / std::fabs(dp(i));
    }
    Eigen::Vector3d t_max;
    for (int i = 0; i < 3; ++i) {
      t_max(i) = step(i) > 0 ? (idx0(i) + 1) - p0(i) / mapPtr_->resolution : p0(i) / mapPtr_->resolution - idx0(i);
    }
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
  }

  inline void visCorridor(const vec_E<Polyhedron3D>& polyhedra) {
    decomp_ros_msgs::PolyhedronArray poly_msg = DecompROS::polyhedron_array_to_ros(polyhedra);
    poly_msg.header.frame_id = "world";
    poly_msg.header.stamp = ros::Time::now();
    hPolyPub_.publish(poly_msg);
  }
  inline void visCorridor(const std::vector<Eigen::MatrixXd>& hPolys) {
    vec_E<Polyhedron3D> decompPolys;
    for (const auto& poly : hPolys) {
      vec_E<Hyperplane3D> hyper_planes;
      hyper_planes.resize(poly.cols());
      for (int i = 0; i < poly.cols(); ++i) {
        hyper_planes[i].n_ = poly.col(i).head(3);
        hyper_planes[i].p_ = poly.col(i).tail(3);
      }
      decompPolys.emplace_back(hyper_planes);
    }
    visCorridor(decompPolys);
  }

};

}