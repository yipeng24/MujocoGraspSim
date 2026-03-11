/*
Copyright (C) 2022 Hongkai Ye (kyle_yeh@163.com)
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
#ifndef _BIAS_SAMPLER_
#define _BIAS_SAMPLER_

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <random>

using namespace std;
template <class VEC>
class BiasSampler
{
public:
  BiasSampler()
  {
    std::random_device rd;
    gen_ = std::mt19937_64(rd());
    uniform_rand_ = std::uniform_real_distribution<double>(0.0, 1.0);
    normal_rand_ = std::normal_distribution<double>(0.0, 1.0);
    range_.setZero();
    origin_.setZero();
    informed_ = false;
    GUILD_informed_ = false;
  };
  
  // template <class VEC>
  void setSamplingRange(const VEC origin, const VEC range)
  {
    origin_ = origin;
    range_ = range;
    std::cout << "[sampler]:origin: " << origin_.transpose() << std::endl;
    std::cout << "[sampler]:range: " << range_.transpose() << std::endl;
  }

  // template <class VEC>
  void samplingOnce(VEC &sample)
  {
    // TODO change dim
    if (informed_)
    {
      informedSamplingOnce(sample);
    }
    // else if (GUILD_informed_)
    // {
    //   GUILDSamplingOnce(sample);
    // }
    // else
    {
      uniformSamplingOnce(sample);
    }
  }

  // template <class VEC>
  void uniformSamplingOnce(VEC &sample)
  {
    // sample[0] = uniform_rand_(gen_);
    // sample[1] = uniform_rand_(gen_);
    // sample[2] = uniform_rand_(gen_);
    VEC sample_yaw(sample.size()-1);
    for (unsigned char i = 0; i < sample_yaw.size(); i++)
      sample_yaw[i] = uniform_rand_(gen_);

    sample_yaw.array() *= range_.array();
    sample_yaw += origin_;

    sample.head(3) = sample_yaw.head(3);
    sample[3] = cos(sample_yaw[3]);
    sample[4] = sin(sample_yaw[3]);
    sample.tail(sample.size()-5) = sample_yaw.tail(sample_yaw.size()-4);
  };


  // TODO change dim
  // template <class VEC>
  void informedSamplingOnce(VEC &sample)
  {
    // 1 pos sample
    Eigen::Vector3d p_sample;
    // random uniform sampling in a unit 3-ball
    Eigen::Vector3d p;
    p[0] = normal_rand_(gen_);
    p[1] = normal_rand_(gen_);
    p[2] = normal_rand_(gen_);
    double r = pow(uniform_rand_(gen_), 0.33333);
    p_sample = r * p.normalized();

    // transform the pt into the ellipsoid
    p_sample.array() *= radii_.array();
    p_sample = rotation_ * p_sample;
    p_sample += center_;

    // 2 others sample
    for (unsigned char i = 3; i < sample.size(); i++)
      sample[i] = uniform_rand_(gen_);
    
    sample.array() *= range_.array();
    sample += origin_;

    sample.head(3) = p_sample;

    double yaw = 2.0 * M_PI * uniform_rand_(gen_);
    sample[3] = cos(yaw);
    sample[4] = sin(yaw);
  }

  // TODO change dim
  // void GUILDSamplingOnce(Eigen::Vector3d &sample)
  // {
  //   // random uniform sampling in a unit 3-ball
  //   Eigen::Vector3d p;
  //   p[0] = normal_rand_(gen_);
  //   p[1] = normal_rand_(gen_);
  //   p[2] = normal_rand_(gen_);
  //   double r = pow(uniform_rand_(gen_), 0.33333);
  //   sample = r * p.normalized();

  //   // transform the pt into the ellipsoid
  //   if (p[0] > 0.0)
  //   {
  //     sample.array() *= radii_s_.array();
  //     sample = rotation_s_ * sample;
  //     sample += center_s_;
  //   }
  //   else
  //   {
  //     sample.array() *= radii_g_.array();
  //     sample = rotation_g_ * sample;
  //     sample += center_g_;
  //   }
  // }

  // TODO change dim
  void setInformedTransRot(const Eigen::Vector3d &trans, const Eigen::Matrix3d &rot)
  {
    center_ = trans;
    rotation_ = rot;
  }

  // TODO change dim
  void setInformedSacling(const Eigen::Vector3d &scale)
  {
    informed_ = true;
    radii_ = scale;
  }

  // TODO change dim
  // void setGUILDInformed(const Eigen::Vector3d &scale1, const Eigen::Vector3d &trans1, const Eigen::Matrix3d &rot1,
  //                       const Eigen::Vector3d &scale2, const Eigen::Vector3d &trans2, const Eigen::Matrix3d &rot2)
  // {
  //   radii_s_ = scale1;
  //   center_s_ = trans1;
  //   rotation_s_ = rot1;
  //   radii_g_ = scale2;
  //   center_g_ = trans2;
  //   rotation_g_ = rot2;
  // }

  void reset()
  {
    informed_ = false;
    GUILD_informed_ = false;
  }

  // (0.0 - 1.0)
  double getUniRandNum()
  {
    return uniform_rand_(gen_);
  }

private:
  // Eigen::Vector3d range_, origin_;
  // // template <class VEC>
  VEC range_, origin_;

  std::mt19937_64 gen_;
  std::uniform_real_distribution<double> uniform_rand_;
  std::normal_distribution<double> normal_rand_;

  // TODO change dim
  // for informed sampling
  bool informed_;
  Eigen::Vector3d center_, radii_;
  Eigen::Matrix3d rotation_;

  // for GUILD informed sampling
  bool GUILD_informed_;
  // Eigen::Vector3d center_s_, radii_s_;
  // Eigen::Vector3d center_g_, radii_g_;
  // Eigen::Matrix3d rotation_s_;
  // Eigen::Matrix3d rotation_g_;
};

#endif
