/*
 * @Author: BbbbigRui bbbbigrui@zju.edu.cn
 * @Date: 2023-03-02 15:55:09
 * @LastEditors: BbbbigRui bbbbigrui@zju.edu.cn
 * @LastEditTime: 2023-04-20 23:58:14
 * @FilePath: /src/uav_simulator/mockamap/include/maps.hpp
 * @Description: 
 * 
 * Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
 */
#ifndef MAPS_HPP
#define MAPS_HPP

#include <ros/ros.h>

#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>

namespace mocka {

struct BOX{
  Eigen::Vector3d size;
  Eigen::Matrix3d R_b2w;
  Eigen::Vector3d T_b2w;
};

class Maps {
public:
  typedef struct BasicInfo {
    ros::NodeHandle *nh_private;
    int sizeX;
    int sizeY;
    int sizeZ;
    int seed;
    double scale;
    sensor_msgs::PointCloud2 *output;
    pcl::PointCloud<pcl::PointXYZ> *cloud;
  } BasicInfo;

  BasicInfo getInfo() const;
  void setInfo(const BasicInfo &value);

public:
  Maps();

public:
  void generate(int type);

private:
  BasicInfo info;
  std::vector<BOX> boxes_;

private:
  void pcl2ros(const bool add_color = false);

  void perlin3D();
  void maze2D();
  void randomMapGenerate();
  //revised by JR
  void gapWallGen();
  void clutterMapGen();
  void onePtMapGen();

  void Maze3DGen();
  void recursiveDivision(int xl, int xh, int yl, int yh, Eigen::MatrixXi &maze);
  void recursizeDivisionMaze(Eigen::MatrixXi &maze);
  void optimizeMap();
  void addGround();

  void eular2rot(const Eigen::Vector3d& ea, Eigen::Matrix3d& R);
  bool inAllBoxes(const Eigen::Vector3d& pt);
  bool inAnyBox(const Eigen::Vector3d& pt);

  void addBox(const Eigen::Matrix3d& R_b2w, const Eigen::Vector3d& T_b2w, const Eigen::Vector3d& size);
};

class MazePoint {
private:
  pcl::PointXYZ point;
  double dist1;
  double dist2;
  int point1;
  int point2;
  bool isdoor;

public:
  pcl::PointXYZ getPoint();
  int getPoint1();
  int getPoint2();
  double getDist1();
  double getDist2();
  void setPoint(pcl::PointXYZ p);
  void setPoint1(int p);
  void setPoint2(int p);
  void setDist1(double set);
  void setDist2(double set);
};

} // namespace mocka

#endif // MAPS_HPP
