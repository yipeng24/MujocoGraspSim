#include "maps.hpp"

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

#include <Eigen/Core>

#include "perlinnoise.hpp"

using namespace mocka;

// @GaaiLam
void Maps::addGround() {
  int add_ground;
  info.nh_private->param("ground", add_ground, 0);
  if (add_ground) {
    double _x_l = -info.sizeX / (2 * info.scale);
    double _y_l = -info.sizeY / (2 * info.scale);
    for (int i=0; i<info.sizeX; ++i) {
      for (int j=0; j<info.sizeY; ++j) {
        info.cloud->points.emplace_back(
          _x_l+i*1.0/info.scale, _y_l+j*1.0/info.scale, 0);
      }
    }
  }
}

void
Maps::randomMapGenerate()
{

  std::default_random_engine eng(info.seed);

  double _resolution = 1 / info.scale;

  double _x_l = -info.sizeX / (2 * info.scale);
  double _x_h = info.sizeX / (2 * info.scale);
  double _y_l = -info.sizeY / (2 * info.scale);
  double _y_h = info.sizeY / (2 * info.scale);
  // double _h_l = 0;
  // double _h_h = info.sizeZ / info.scale;

  double _w_l, _w_h, _h_l, _h_h;
  int    _ObsNum;

  info.nh_private->param("width_min", _w_l, 0.6);
  info.nh_private->param("width_max", _w_h, 1.5);
  info.nh_private->param("height_min", _h_l, 1.5);
  info.nh_private->param("height_max", _h_h, 1.5);
  info.nh_private->param("obstacle_number", _ObsNum, 10);

  _h_l = _h_l >= 0 ? _h_l : 0;
  _h_h = _h_h <= info.sizeZ / info.scale ? _h_h : info.sizeZ / info.scale;

  std::uniform_real_distribution<double> rand_x;
  std::uniform_real_distribution<double> rand_y;
  std::uniform_real_distribution<double> rand_w;
  std::uniform_real_distribution<double> rand_h;

  pcl::PointXYZ pt_random;

  rand_x = std::uniform_real_distribution<double>(_x_l, _x_h);
  rand_y = std::uniform_real_distribution<double>(_y_l, _y_h);
  rand_w = std::uniform_real_distribution<double>(_w_l, _w_h);
  rand_h = std::uniform_real_distribution<double>(_h_l, _h_h);

  for (int i = 0; i < _ObsNum; i++)
  {
    double x, y;
    x = rand_x(eng);
    y = rand_y(eng);

    double w, h;
    w = rand_w(eng);
    h = rand_h(eng);

    int widNum = ceil(w / _resolution);
    int heiNum = ceil(h / _resolution);

    int rl, rh, sl, sh;
    rl = -widNum / 2;
    rh = widNum / 2;
    sl = -widNum / 2;
    sh = widNum / 2;

    for (int r = rl; r < rh; r++)
      for (int s = sl; s < sh; s++)
      {
        for (int t = 0; t < heiNum; t++)
        {
          // if(r>=rl && r<=rh-1 && s>=sl && s<=sh-1 && t>=0 && t<=heiNum-1)
          if ((r - rl) * (r - rh + 1) * (s - sl) * (s - sh + 1) * t *
                (t - heiNum + 1) == 0)
          {
            pt_random.x = x + r * _resolution + 0.5 * _resolution;
            pt_random.y = y + s * _resolution + 0.5 * _resolution;
            pt_random.z = t * _resolution + 0.5 * _resolution;
            info.cloud->points.push_back(pt_random);
          }
        }
      }
  }
  addGround();
  info.cloud->width    = info.cloud->points.size();
  info.cloud->height   = 1;
  info.cloud->is_dense = true;
  pcl2ros();
}
                      //0 1 2 对应 z y x
void Maps::eular2rot(const Eigen::Vector3d& ea, Eigen::Matrix3d& R){
    R = Eigen::AngleAxisd(ea[0], Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(ea[1], Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(ea[2], Eigen::Vector3d::UnitX());
}


void Maps::addBox(const Eigen::Matrix3d& R_b2w, const Eigen::Vector3d& T_b2w, 
                  const Eigen::Vector3d& size){
  BOX box;
  box.R_b2w = R_b2w;
  box.T_b2w = T_b2w;
  box.size = size;
  boxes_.push_back(box);
}

bool Maps::inAllBoxes(const Eigen::Vector3d& pt_w){
  // std::cout << "pt_w:" << pt_w.transpose() << std::endl;
  for(auto& box:boxes_){
    Eigen::Vector3d pt_b = box.R_b2w.transpose()*(pt_w-box.T_b2w);
    // std::cout << "pt_b:" << pt_b.transpose() << std::endl;
    for(int i=0;i<3;++i){
      if(fabs(pt_b(i))>0.5*box.size(i)){
        // std::cout << "true" << std::endl;
        return false;
      }
    }
  }
  // std::cout << "false" << std::endl;
  return true;
}

bool Maps::inAnyBox(const Eigen::Vector3d& pt_w){
  for(auto& box:boxes_){
    Eigen::Vector3d pt_b = box.R_b2w.transpose()*(pt_w-box.T_b2w);

    bool in_box = true;
    for(int i=0;i<3;++i){
      if(fabs(pt_b(i))>=0.5*box.size(i)){
        in_box = false;
        break;
      }
    }
    if(in_box) return true;
  }

  return false;
}

// revised by JR
void
Maps::gapWallGen()
{

  double box0_size_x, box0_size_y, box0_size_z;
  info.nh_private->param("box0_size_x", box0_size_x, 0.6);
  info.nh_private->param("box0_size_y", box0_size_y, 1.5);
  info.nh_private->param("box0_size_z", box0_size_z, 1.5);

  Eigen::Matrix3d R_b2w;
  Eigen::Vector3d T_b2w,size,ea;

  ea<<0.,0.,1.;
  eular2rot(ea,R_b2w);
  T_b2w<<0.,0.,1.5;
  size<<box0_size_x,box0_size_y,box0_size_z;
  addBox(R_b2w,T_b2w,size);

  double _resolution = 1 / info.scale;

  pcl::PointXYZ pt_random;
  Eigen::Vector3d pt;

  for(int i= -info.sizeX / 2;i<info.sizeX / 2;++i){
    for(int j= -info.sizeY / 2;j<info.sizeY / 2;++j){
      for(int k= 0;k<info.sizeZ;++k){

        // QUAD
        // if(i==0 && (k<1.2*j+10 || k>1.2*j+25 || k<5 || k>20)){

        // COAX
        // double c,c_inv;
        // c = -1.2;
        // c_inv = -1.0/c;
        // int r=1.2*info.scale;
        // int o_z = 2.5*info.scale;

        //共轴斜着的墙
        // if((i==0 || i==2) && !((k>c*j+o_z-r && k<c*j+o_z+r && k<c_inv*j+o_z && k>c_inv*j+o_z-0.5*r)||  // hub
        //              (k>=c*j+o_z-0.3*r && k<=c*j+o_z+0.3*r && k>=c_inv*j+o_z-1.2*r &&  k<=c_inv*j+o_z-0.5*r))){ // rotor

        // if((i==0 || i==2) && !(k>=c*j+o_z-0.3*r && k<=c*j+o_z+0.3*r && k>=c_inv*j+o_z-1.2*r &&  k<=c_inv*j+o_z-0.5*r)){
        //   pt_random.x = (i+0.5)*_resolution;
        //   pt_random.y = (j+0.5)*_resolution;
        //   pt_random.z = (k+0.5)*_resolution;
        //   info.cloud->points.push_back(pt_random);
        // }

        pt.x() = (i+0.5)*_resolution;
        pt.y() = (j+0.5)*_resolution;
        pt.z() = (k+0.5)*_resolution;
        if((i==0 || i==2) && !inAllBoxes(pt)){
          pt_random.x = pt.x();
          pt_random.y = pt.y();
          pt_random.z = pt.z();
          info.cloud->points.push_back(pt_random);
        }

  }}}

  addGround();
  info.cloud->width    = info.cloud->points.size();
  info.cloud->height   = 1;
  info.cloud->is_dense = true;
  pcl2ros();
}

void Maps::onePtMapGen()
{
  Eigen::Vector3d pt;
  info.nh_private->param("pt_x", pt.x(), 0.0);
  info.nh_private->param("pt_y", pt.y(), 0.0);
  info.nh_private->param("pt_z", pt.z(), 0.0);

  pcl::PointXYZ pcl;
  pcl.x = pt.x();
  pcl.y = pt.y();
  pcl.z = pt.z();

  info.cloud->points.push_back(pcl);

  info.cloud->width    = info.cloud->points.size();
  info.cloud->height   = 1;
  info.cloud->is_dense = true;
  pcl2ros();

}

// revised by JR
void Maps::clutterMapGen()
{
  std::cout << "clutterMapGen" << std::endl;
  std::default_random_engine eng(info.seed);

  double _resolution = 1 / info.scale;
  double _x_l = -info.sizeX / (2 * info.scale);
  double _x_h = info.sizeX / (2 * info.scale);
  double _y_l = -info.sizeY / (2 * info.scale);
  double _y_h = info.sizeY / (2 * info.scale);
  double _z_l = 0;
  double _z_h = info.sizeZ / info.scale;

  double min_len, max_len, min_wid, max_wid;
  info.nh_private->param("min_len", min_len, 0.6);
  info.nh_private->param("max_len", max_len, 1.5);
  info.nh_private->param("min_wid", min_wid, 0.3);
  info.nh_private->param("max_wid", max_wid, 0.6);

  int    _ObsNum;
  info.nh_private->param("obs_num", _ObsNum, 10);
  std::cout << "min_len: " << min_len << std::endl;
  std::cout << "max_len: " << max_len << std::endl;
  std::cout << "min_wid: " << min_wid << std::endl;
  std::cout << "max_wid: " << max_wid << std::endl;
  std::cout << "obs_num: " << _ObsNum << std::endl;

  std::uniform_real_distribution<double> rand_x;
  std::uniform_real_distribution<double> rand_y;
  std::uniform_real_distribution<double> rand_z;
  std::uniform_real_distribution<double> rand_len;
  std::uniform_real_distribution<double> rand_yaw;
  std::uniform_real_distribution<double> rand_pitch;
  std::uniform_real_distribution<double> rand_wid;

  rand_x = std::uniform_real_distribution<double>(_x_l, _x_h);
  rand_y = std::uniform_real_distribution<double>(_y_l, _y_h);
  rand_z = std::uniform_real_distribution<double>(_z_l, _z_h);
  rand_len = std::uniform_real_distribution<double>(min_len, max_len);
  rand_yaw = std::uniform_real_distribution<double>(0, 2 * M_PI);
  rand_pitch = std::uniform_real_distribution<double>(-M_PI / 2, M_PI / 2);
  rand_wid = std::uniform_real_distribution<double>(min_wid, max_wid);

  for (int i = 0; i < _ObsNum; i++)
  {
    double x, y, z, len, yaw, pitch, wid;
    x = rand_x(eng);
    y = rand_y(eng);
    z = rand_z(eng);
    len = rand_len(eng);
    yaw = rand_yaw(eng);
    pitch = rand_pitch(eng);
    wid = rand_wid(eng);

    Eigen::Matrix3d R_b2w;
    Eigen::Vector3d T_b2w,size,ea;
    T_b2w<<x,y,z;
    ea<<yaw, pitch, 0;
    eular2rot(ea,R_b2w);



    int widNum = ceil(wid / _resolution);
    int lenNum = ceil(len / _resolution);

    int rl, rh, sl, sh;
    rl = -widNum / 2;
    rh = widNum / 2;
    sl = -widNum / 2;
    sh = widNum / 2;
    
    pcl::PointXYZ pt_random;

    for (int r = rl; r < rh; r++)
      for (int s = sl; s < sh; s++)
        for (int t = 0; t < lenNum; t++)
          if ((r - rl) * (r - rh + 1) * (s - sl) 
              * (s - sh + 1) * t * (t - lenNum + 1) == 0)
          {
            Eigen::Vector3d pt;
            pt.x() = r * _resolution + 0.5 * _resolution;
            pt.y() = s * _resolution + 0.5 * _resolution;
            pt.z() = t * _resolution + 0.5 * _resolution;
            
            // 变化pt，并落在grid
            pt = R_b2w * pt + T_b2w;
            Eigen::Vector3i idx;
            idx << floor(pt.x() * info.scale),
                  floor(pt.y() * info.scale),
                  floor(pt.z() * info.scale);

            pt << (idx.x()+0.5)*_resolution,
                (idx.y()+0.5)*_resolution,
                (idx.z()+0.5)*_resolution;

            pt_random.x = pt.x();
            pt_random.y = pt.y();
            pt_random.z = pt.z();
            info.cloud->points.push_back(pt_random);
          }
  }
  
  std::cout << "pts num: " << info.cloud->points.size() << std::endl;

  addGround();
  info.cloud->width    = info.cloud->points.size();
  info.cloud->height   = 1;
  info.cloud->is_dense = true;
  pcl2ros();
}

void
Maps::pcl2ros(const bool add_color)
{
  if(add_color)
  {
    pcl::PointCloud<pcl::PointXYZRGB> cloud_rgb;
    cloud_rgb.width = info.cloud->width;
    cloud_rgb.height = 1;
    cloud_rgb.points.resize(cloud_rgb.width * cloud_rgb.height);
    for (int i = 0; i < cloud_rgb.points.size(); ++i)
    {
      cloud_rgb.points[i].x = info.cloud->points[i].x;
      cloud_rgb.points[i].y = info.cloud->points[i].y;
      cloud_rgb.points[i].z = info.cloud->points[i].z;

      cloud_rgb.points[i].r = int (255 * fabs(cos(info.cloud->points[i].x)));
      cloud_rgb.points[i].g = int (255 * fabs(sin(0.5+0.2*info.cloud->points[i].y)));
      cloud_rgb.points[i].b = 100;
    }
    pcl::toROSMsg(cloud_rgb, *info.output);
  }
  else
    pcl::toROSMsg(*info.cloud, *info.output);
  
  info.output->header.frame_id = "world";
  ROS_INFO("finish: infill %lf%%",
           info.cloud->width / (1.0 * info.sizeX * info.sizeY * info.sizeZ));
}

void
Maps::perlin3D()
{
  double complexity;
  double fill;
  int    fractal;
  double attenuation;

  info.nh_private->param("complexity", complexity, 0.142857);
  info.nh_private->param("fill", fill, 0.38);
  info.nh_private->param("fractal", fractal, 1);
  info.nh_private->param("attenuation", attenuation, 0.5);

  info.cloud->width  = info.sizeX * info.sizeY * info.sizeZ;
  info.cloud->height = 1;
  info.cloud->points.resize(info.cloud->width * info.cloud->height);

  PerlinNoise noise(info.seed);

  std::vector<double>* v = new std::vector<double>;
  v->reserve(info.cloud->width);
  for (int i = 0; i < info.sizeX; ++i)
  {
    for (int j = 0; j < info.sizeY; ++j)
    {
      for (int k = 0; k < info.sizeZ; ++k)
      {
        double tnoise = 0;
        for (int it = 1; it <= fractal; ++it)
        {
          int    dfv = pow(2, it);
          double ta  = attenuation / it;
          tnoise += ta * noise.noise(dfv * i * complexity,
                                     dfv * j * complexity,
                                     dfv * k * complexity);
        }
        v->push_back(tnoise);
      }
    }
  }
  std::sort(v->begin(), v->end());
  int    tpos = info.cloud->width * (1 - fill);
  double tmp  = v->at(tpos);
  ROS_INFO("threshold: %lf", tmp);

  int pos = 0;
  for (int i = 0; i < info.sizeX; ++i)
  {
    for (int j = 0; j < info.sizeY; ++j)
    {
      for (int k = 0; k < info.sizeZ; ++k)
      {
        double tnoise = 0;
        for (int it = 1; it <= fractal; ++it)
        {
          int    dfv = pow(2, it);
          double ta  = attenuation / it;
          tnoise += ta * noise.noise(dfv * i * complexity,
                                     dfv * j * complexity,
                                     dfv * k * complexity);
        }
        if (tnoise > tmp)
        {
          info.cloud->points[pos].x =
            i / info.scale - info.sizeX / (2 * info.scale) + 0.5 / info.scale;
          info.cloud->points[pos].y =
            j / info.scale - info.sizeY / (2 * info.scale) + 0.5 / info.scale;
          info.cloud->points[pos].z = k / info.scale;
          pos++;
        }
      }
    }
  }
  info.cloud->width = pos;
  ROS_INFO("the number of points before optimization is %d", info.cloud->width);
  info.cloud->points.resize(info.cloud->width * info.cloud->height);
  addGround();
  info.cloud->width = info.cloud->points.size();
  pcl2ros();
}

void
Maps::recursiveDivision(int xl, int xh, int yl, int yh, Eigen::MatrixXi& maze)
{
  ROS_INFO(
    "generating maze with width %d , height %d", xh - xl + 1, yh - yl + 1);

  if (xl < xh - 3 && yl < yh - 3)
  { // the remaining area is larger than or equal to 5*5, need to add both x
    // wall and y wall
    bool valid = false; // used to judge whether the wall selection is valid
    int  xm    = 0;
    int  ym    = 0;
    ROS_INFO("entered 5*5 mode");
    while (valid == false)
    {
      xm = (std::rand() % (xh - xl - 1) + xl +
            1); // generating random number between xl+1 and xh-1(pointless to
                // add a wall at the sides)
      ym = (std::rand() % (yh - yl - 1) + yl +
            1); // generating random number between yl+1 and yh-1(pointless to
                // add a wall at the sides)
      if (xl - 1 >= 0)
      { // there is a point at xl-1,ym
        if (maze(xl - 1, ym) == 0)
        { // this is an opening,need to change random number
          continue;
        }
      }

      else if (xh + 1 <= maze.cols() - 1)
      { // there is a point at xh+1,ym
        if (maze(xh + 1, ym) == 0)
        { // this is an opening,need to change random number
          continue;
        }
      }

      else if (yl - 1 >= 0)
      { // there is a point at xm,yl-1
        if (maze(xm, yl - 1) == 0)
        { // this is an opening,need to change random number
          continue;
        }
      }

      else if (yh + 1 <= maze.rows() - 1)
      { // there is a point at xm,yh+1
        if (maze(xm, yh + 1) == 0)
        { // this is an opening,need to change random number
          continue;
        }
      }

      valid = true;

    } // xm and ym are now the valid coordinate of the center of the wall
    for (int i = xl; i <= xh; i++)
    {
      maze(i, ym) = 1;
    }
    for (int j = yl; j <= yh; j++)
    {
      maze(xm, j) = 1;
    } // adding walls around the center point
    int d1 = std::rand() % (xm - xl) + xl;
    int d2 = std::rand() % (xh - xm) + xm + 1;
    int d3 = std::rand() % (ym - yl) + yl;
    int d4 =
      std::rand() % (yh - ym) + ym + 1; // generating four possible door points

    int decision = std::rand() % 4; // random selection of three doors
    switch (decision)
    {
      case 0:
        maze(d1, ym) = 0;
        maze(d2, ym) = 0;
        maze(xm, d3) = 0;
        break;

      case 1:
        maze(d1, ym) = 0;
        maze(d2, ym) = 0;
        maze(xm, d4) = 0;
        break;

      case 2:
        maze(d2, ym) = 0;
        maze(xm, d3) = 0;
        maze(xm, d4) = 0;
        break;

      case 3:
        maze(d1, ym) = 0;
        maze(xm, d3) = 0;
        maze(xm, d4) = 0;
        break;
    } // the doors are opened for this cell
    if (yl - 1 >= 0)
    {
      if (maze(xm, yl - 1) == 0)
      {
        maze(xm, yl) = 0;
      }
    }

    if (yh + 1 <= maze.rows() - 1)
    {
      if (maze(xm, yh + 1) == 0)
      {
        maze(xm, yh) = 0;
      }
    }

    if (xl - 1 >= 0)
    {
      if (maze(xl - 1, ym) == 0)
      {
        maze(xl, ym) = 0;
      }
    }

    if (xh + 1 <= maze.cols() - 1)
    {
      if (maze(xh + 1, ym) == 0)
      {
        maze(xh, ym) = 0;
      }
    }

    std::cout << maze << std::endl;
    recursiveDivision(xl, xm - 1, yl, ym - 1, maze);
    recursiveDivision(xm + 1, xh, yl, ym - 1, maze);
    recursiveDivision(xl, xm - 1, ym + 1, yh, maze);
    recursiveDivision(xm + 1, xh, ym + 1, yh, maze);

    ROS_INFO("finished generating maze with width %d , height %d",
             xh - xl + 1,
             yh - yl + 1);
    std::cout << maze << std::endl;
    return;
  } // when the remaining area is larger than or equal to 5*5

  else if (xl < xh - 2 && yl < yh - 2)
  {
    bool valid     = false; // used to judge whether the wall selection is valid
    int  xm        = 0;
    int  ym        = 0;
    int  doorcount = 0;
    xm             = (std::rand() % (xh - xl - 1) + xl +
          1); // generating random number between xl+1 and xh-1(pointless to
                          // add a wall at the sides)
    ym =
      (std::rand() % (yh - yl - 1) + yl +
       1); // generating random number between yl+1 and yh-1(pointless to
           // add a wall at the sides)
           // xm and ym are now the valid coordinate of the center of the wall
    for (int i = xl; i <= xh; i++)
    {
      maze(i, ym) = 1;
    }
    for (int j = yl; j <= yh; j++)
    {
      maze(xm, j) = 1;
    } // adding walls around the center point
    if (yl - 1 >= 0)
    {
      if (maze(xm, yl - 1) == 0)
      {
        maze(xm, yl) = 0;
        doorcount++;
      }
    }

    if (yh + 1 <= maze.rows() - 1)
    {
      if (maze(xm, yh + 1) == 0)
      {
        maze(xm, yh) = 0;
        doorcount++;
      }
    }

    if (xl - 1 >= 0)
    {
      if (maze(xl - 1, ym) == 0)
      {
        maze(xl, ym) = 0;
        doorcount++;
      }
    }

    if (xh + 1 <= maze.cols() - 1)
    {
      if (maze(xh + 1, ym) == 0)
      {
        maze(xh, ym) = 0;
        doorcount++;
      }
    }

    int d1 = std::rand() % (xm - xl) + xl;
    int d2 = std::rand() % (xh - xm) + xm + 1;
    int d3 = std::rand() % (ym - yl) + yl;
    int d4 =
      std::rand() % (yh - ym) + ym + 1; // generating four possible door points

    int decision = std::rand() % 4; // random selection of three doors
    switch (decision)
    {
      case 0:
        maze(d1, ym) = 0;
        maze(d2, ym) = 0;
        maze(xm, d3) = 0;
        break;

      case 1:
        maze(d1, ym) = 0;
        maze(d2, ym) = 0;
        maze(xm, d4) = 0;
        break;

      case 2:
        maze(d2, ym) = 0;
        maze(xm, d3) = 0;
        maze(xm, d4) = 0;
        break;

      case 3:
        maze(d1, ym) = 0;
        maze(xm, d3) = 0;
        maze(xm, d4) = 0;
        break;
    } // the doors are opened for this cell
    std::cout << maze << std::endl;

    ROS_INFO("finished generating maze with width %d , height %d",
             xh - xl + 1,
             yh - yl + 1);
    std::cout << maze << std::endl;
    return;
  }

  else if (xl < xh - 1 && yl < yh - 2)
  { // the case of 3*4+
    ROS_INFO("entered 3*4+ mode");
    int doorcount = 0;
    int ym        = 0;
    for (int i = yl; i <= yh; i++)
    {
      maze(xl + 1, i) = 1;
    } // filling a center wall
    if (yl - 1 >= 0)
    {
      if (maze(xl + 1, yl - 1) == 0)
      {
        maze(xl + 1, yl) = 0;
        doorcount++;
      }
    }
    if (yh + 1 <= maze.rows() - 1)
    {
      if (maze(xl + 1, yh + 1) == 0)
      {
        maze(xl + 1, yh) = 0;
        doorcount++;
      }
    } // opening doors if the wall blocks the old doors
    if (doorcount == 0)
    {
      ym               = std::rand() % (yh - yl + 1) + yl;
      maze(xl + 1, ym) = 0;
    }
  } // the case of 4+*3
  //
  else if (xl < xh - 2 && yl < yh - 1)
  { // the case of 4+*3
    ROS_INFO("entered 4+*3 mode");
    int doorcount = 0;
    int xm        = 0;
    for (int i = xl; i <= xh; i++)
    {
      maze(i, yl + 1) = 1;
    } // filling a center wall
    if (xl - 1 >= 0)
    {
      if (maze(xl - 1, yl + 1) == 0)
      {
        maze(xl, yl + 1) = 0;
        doorcount++;
      }
    }
    if (xh + 1 <= maze.cols() - 1)
    {
      if (maze(xh + 1, yl + 1) == 0)
      {
        maze(xh, yl + 1) = 0;
        doorcount++;
      }
    } // opening doors if the wall blocks the old doors
    if (doorcount == 0)
    {
      xm               = std::rand() % (xh - xl + 1) + xl;
      maze(xm, yl + 1) = 0;
    }
  } // the case of 4+*3

  else if (xl < xh - 1 && yl < yh - 1)
  { // the case of 3*3
    maze(xl + 1, yl + 1) = 1;
    return;
  }
  else
  {
    ROS_INFO("finished generating maze with width %d , height %d",
             xh - xl + 1,
             yh - yl + 1);
    return;
  }
}

void
Maps::recursizeDivisionMaze(Eigen::MatrixXi& maze)
{
  //! @todo all bugs here...
  int sx = maze.rows();
  int sy = maze.cols();

  int px, py;

  if (sx > 5)
    px = (std::rand() % (sx - 3) + 1);
  else
    return;

  if (sy > 5)
    py = (std::rand() % (sy - 3) + 1);
  else
    return;

  ROS_INFO("debug %d %d %d %d", sx, sy, px, py);

  int x1, x2, y1, y2;

  if (px != 1)
    x1 = (std::rand() % (px - 1) + 1);
  else
    x1 = 1;

  if ((sx - px - 3) > 0)
    x2 = (std::rand() % (sx - px - 3) + px + 1);
  else
    x2 = px + 1;

  if (py != 1)
    y1 = (std::rand() % (py - 1) + 1);
  else
    y1 = 1;

  if ((sy - py - 3) > 0)
    y2 = (std::rand() % (sy - py - 3) + py + 1);
  else
    y2 = py + 1;
  ROS_INFO("%d %d %d %d", x1, x2, y1, y2);

  if (px != 1 && px != (sx - 2))
  {
    for (int i = 1; i < (sy - 1); ++i)
    {
      if (i != y1 && i != y2)
        maze(px, i) = 1;
    }
  }
  if (py != 1 && py != (sy - 2))
  {
    for (int i = 1; i < (sx - 1); ++i)
    {
      if (i != x1 && i != x2)
        maze(i, py) = 1;
    }
  }
  switch (std::rand() % 4)
  {
    case 0:
      maze(x1, py) = 1;
      break;
    case 1:
      maze(x2, py) = 1;
      break;
    case 2:
      maze(px, y1) = 1;
      break;
    case 3:
      maze(px, y2) = 1;
      break;
  }

  if (px > 2 && py > 2)
  {
    Eigen::MatrixXi sub = maze.block(0, 0, px + 1, py + 1);
    recursizeDivisionMaze(sub);
    maze.block(0, 0, px, py) = sub;
  }
  if (px > 2 && (sy - py - 1) > 2)
  {
    Eigen::MatrixXi sub = maze.block(0, py, px + 1, sy - py);
    recursizeDivisionMaze(sub);
    maze.block(0, py, px + 1, sy - py) = sub;
  }
  if (py > 2 && (sx - px - 1) > 2)
  {
    Eigen::MatrixXi sub = maze.block(px, 0, sx - px, py + 1);
    recursizeDivisionMaze(sub);
    maze.block(px, 0, sx - px, py + 1) = sub;
  }
  if ((sx - px - 1) > 2 && (sy - py - 1) > 2)
  {

    Eigen::MatrixXi sub = maze.block(px, py, sy - px, sy - py);

    recursizeDivisionMaze(sub);
    maze.block(px, py, sy - px, sy - py) = sub;
  }
}

void
Maps::maze2D()
{
  double width;
  int    type;
  int    addWallX;
  int    addWallY;
  info.nh_private->param("road_width", width, 1.0);
  info.nh_private->param("add_wall_x", addWallX, 0);
  info.nh_private->param("add_wall_y", addWallY, 0);
  info.nh_private->param("maze_type", type, 1);

  int mx = info.sizeX / (width * info.scale);
  int my = info.sizeY / (width * info.scale);

  Eigen::MatrixXi maze(mx, my);
  maze.setZero();

  switch (type)
  {
    case 1:
      recursiveDivision(0, maze.cols() - 1, 0, maze.rows() - 1, maze);
      break;
  }

  if (addWallX)
  {
    for (int i = 0; i < mx; ++i)
    {
      maze(i, 0)      = 1;
      maze(i, my - 1) = 1;
    }
  }
  if (addWallY)
  {
    for (int i = 0; i < my; ++i)
    {
      maze(0, i)      = 1;
      maze(mx - 1, i) = 1;
    }
  }

  std::cout << maze << std::endl;

  for (int i = 0; i < mx; ++i)
  {
    for (int j = 0; j < my; ++j)
    {
      if (maze(i, j))
      {
        for (int ii = 0; ii < width * info.scale; ++ii)
        {
          for (int jj = 0; jj < width * info.scale; ++jj)
          {
            for (int k = 0; k < info.sizeZ; ++k)
            {
              if(ii == 0 || ii == (width * info.scale - 1) 
                 || jj == 0 || jj == (width * info.scale - 1)
                 || k == 0 || k == (info.sizeZ - 1))
              {
                pcl::PointXYZ pt_random;
                pt_random.x =
                  i * width + ii / info.scale - info.sizeX / (2.0 * info.scale);
                pt_random.y =
                  j * width + jj / info.scale - info.sizeY / (2.0 * info.scale);
                pt_random.z = k / info.scale;
                info.cloud->points.push_back(pt_random);
              }
            }
          }
        }
      }
    }
  }
  addGround();
  info.cloud->width    = info.cloud->points.size();
  info.cloud->height   = 1;
  info.cloud->is_dense = true;
  pcl2ros(true);
}

Maps::BasicInfo
Maps::getInfo() const
{
  return info;
}

void
Maps::setInfo(const BasicInfo& value)
{
  info = value;
}

Maps::Maps()
{
}

void
Maps::generate(int type)
{
  switch (type)
  {
    default:
    case 1:
      perlin3D();
      break;
    case 2:
      randomMapGenerate();
      break;
    case 3:
      std::srand(info.seed);
      maze2D();
      break;
    case 4: // generating 3d maze
      std::cout << "generating 3d maze" << std::endl;
      std::srand(info.seed);
      Maze3DGen();

      break;
    case 5:
      std::srand(info.seed);
      gapWallGen();
      break;

    case 6:
      std::cout << "generating random clutter map" << std::endl;
      std::srand(info.seed);
      clutterMapGen();
      break;

    case 7:
      std::cout << "generating one_point map" << std::endl;
      onePtMapGen();
      break;

  }
}

pcl::PointXYZ
MazePoint::getPoint()
{
  return point;
}

int
MazePoint::getPoint1()
{
  return point1;
}

int
MazePoint::getPoint2()
{
  return point2;
}

double
MazePoint::getDist1()
{
  return dist1;
}

double
MazePoint::getDist2()
{
  return dist2;
}

void
MazePoint::setPoint(pcl::PointXYZ p)
{
  point = p;
}

void
MazePoint::setPoint1(int p)
{
  point1 = p;
}

void
MazePoint::setPoint2(int p)
{
  point2 = p;
}

void
MazePoint::setDist1(double set)
{
  dist1 = set;
}

void
MazePoint::setDist2(double set)
{
  dist2 = set;
}

void
Maps::Maze3DGen()
{
  // getting required info parameters from the given node
  int    numNodes;
  double connectivity;
  int    nodeRad;
  int    roadRad;

  std::cout << "generating 3d maze" << std::endl;
  info.nh_private->param("numNodes", numNodes, 10);
  info.nh_private->param("connectivity", connectivity, 0.5);
  info.nh_private->param("nodeRad", nodeRad, 3);
  info.nh_private->param("roadRad", roadRad, 2);
  ROS_INFO("received parameters : numNodes: %d connectivity: "
           "%f nodeRad: %d roadRad: %d",
           numNodes,
           connectivity,
           nodeRad,
           roadRad);
  // generating random points
  std::vector<pcl::PointXYZ> base;

  for (int i = 0; i < numNodes; i++)
  {
    double rx = std::rand() / RAND_MAX +
                (std::rand() % info.sizeX) / info.scale -
                info.sizeX / (2 * info.scale);
    double ry = std::rand() / RAND_MAX +
                (std::rand() % info.sizeY) / info.scale -
                info.sizeY / (2 * info.scale);
    double rz = std::rand() / RAND_MAX +
                (std::rand() % info.sizeZ) / info.scale -
                info.sizeZ / (2 * info.scale);
    ROS_INFO("point: x: %f , y: %f , z: %f", rx, ry, rz);

    pcl::PointXYZ pt_random;
    pt_random.x = rx;
    pt_random.y = ry;
    pt_random.z = rz;
    base.push_back(pt_random);
  } // generating random cores in the space

  for (int i = 0; i < info.sizeX; i++)
  {
    for (int j = 0; j < info.sizeY; j++)
    {
      for (int k = 0; k < info.sizeZ; k++)
      { // for every scaled coordinate points
        pcl::PointXYZ test;
        test.x = i / info.scale - info.sizeX / (2 * info.scale);
        test.y = j / info.scale - info.sizeY / (2 * info.scale);
        test.z = k / info.scale -
                 info.sizeZ /
                   (2 * info.scale); // marking the corresponding point location

        MazePoint mp;
        mp.setPoint(test);
        mp.setPoint2(-1);
        mp.setPoint1(-1);
        mp.setDist1(10000.0);
        mp.setDist2(100000.0); // setting super large starting values
        for (int ii = 0; ii < numNodes; ii++)
        {
          double dist =
            std::sqrt((base[ii].x - test.x) * (base[ii].x - test.x) +
                      (base[ii].y - test.y) * (base[ii].y - test.y) +
                      (base[ii].z - test.z) * (base[ii].z - test.z));
          if (dist < mp.getDist1())
          {

            mp.setDist2(mp.getDist1());
            mp.setDist1(dist);

            mp.setPoint2(mp.getPoint1());
            mp.setPoint1(ii);
          }
          else if (dist < mp.getDist2())
          {
            mp.setDist2(dist);
            mp.setPoint2(ii);
          } // finding the distances to the nearest two cores
        }
        if (std::abs(mp.getDist2() - mp.getDist1()) < 1 / info.scale)
        { // the tested location is on one of the middle planes
          if ((mp.getPoint1() + mp.getPoint2()) >
                int((1 - connectivity) * numNodes) &&
              (mp.getPoint1() + mp.getPoint2()) <
                int((1 + connectivity) * numNodes))
          { // this is a holed wall
            double judge =
              std::sqrt((base[mp.getPoint1()].x - base[mp.getPoint2()].x) *
                          (base[mp.getPoint1()].x - base[mp.getPoint2()].x) +
                        (base[mp.getPoint1()].y - base[mp.getPoint2()].y) *
                          (base[mp.getPoint1()].y - base[mp.getPoint2()].y) +
                        (base[mp.getPoint1()].z - base[mp.getPoint2()].z) *
                          (base[mp.getPoint1()].z - base[mp.getPoint2()].z));
            if (mp.getDist1() + mp.getDist2() - judge >=
                roadRad / (info.scale * 3))
            {
              info.cloud->points.push_back(mp.getPoint());
            }
          }
          else
          {
            info.cloud->points.push_back(mp.getPoint());
          }
        }
      }
    }
  }

  info.cloud->width  = info.cloud->points.size();
  info.cloud->height = 1;
  ROS_INFO("the number of points before optimization is %d", info.cloud->width);
  info.cloud->points.resize(info.cloud->width * info.cloud->height);
  pcl2ros();
}
