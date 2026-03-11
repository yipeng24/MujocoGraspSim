#include <math.h>
#include <vector>

#include <ros/ros.h>
#include <sensor_msgs/Joy.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <mavros_msgs/RCIn.h>

namespace fakerc 
{
  static constexpr uint32_t Mode = 4; // up = -1.0, mid = 0.0, down = 1.0
  static constexpr uint32_t Gear = 5; // up = -1.0, down = 1.0 
  static constexpr uint32_t W1 = 3;   // yaw_channel, left = 1.0, right = -1.0
  static constexpr uint32_t W2 = 1;   // pitch_channel, up = -1.0, down = 1.0
  static constexpr uint32_t W3 = 0;   // roll_channel, left = 1.0, right = -1.0
}

class FakeRC 
{
  private:
    ros::NodeHandle nh_, pnh_;
    ros::Subscriber fakerc_sub;
    ros::Subscriber ch5_sub;
    ros::Subscriber ch6_sub;
    ros::Publisher fakerc_pub;
    ros::WallTimer loop_timer;
    sensor_msgs::Joy fakerc_cmd;
    ros::Time last_msg_time;

    double joystick_timeout_ = 0.5;
    double zero_tolerance = 0.05;
    bool use_joy = false;
    bool use_sim = false;

    bool in_mode_ = false;
    bool in_gear_ = false;

  public:
    FakeRC(const ros::NodeHandle& nh, const ros::NodeHandle& pnh):
        nh_(nh), pnh_(pnh),last_msg_time()
    {
      fakerc_cmd = sensor_msgs::Joy();
      fakerc_cmd.axes = std::vector<float>(8, 0);
      fakerc_cmd.buttons = std::vector<int32_t>(8, 0);

      pnh_.getParam("fake_rc/zero_tolerance", zero_tolerance);
      pnh_.getParam("fake_rc/use_joy", use_joy);
      pnh_.getParam("fake_rc/use_sim", use_sim);

      fakerc_pub = nh_.advertise<mavros_msgs::RCIn>("rcin", 100);

      if (use_joy)
      {
        fakerc_sub = nh_.subscribe("joy", 10, &FakeRC::joyCallback, this);
        loop_timer = nh_.createWallTimer(ros::WallDuration(0.1), &FakeRC::mainLoop, this);
      }
      else
      {
        ch5_sub = nh_.subscribe("/initialpose", 10, &FakeRC::ch5CallBack, this);
        ch6_sub = nh_.subscribe("/clicked_point", 10, &FakeRC::ch6CallBack, this);
        // updateModeChannel();
        // updateGearChannel();
        loop_timer = nh_.createWallTimer(ros::WallDuration(0.1), &FakeRC::rvizLoop, this);
      }

    }

    FakeRC():FakeRC(ros::NodeHandle(),ros::NodeHandle("~")){ }

    void joyCallback(const sensor_msgs::Joy::ConstPtr& msg) 
    {
      fakerc_cmd = *msg;
      last_msg_time = ros::Time::now();

      if (fabs(fakerc_cmd.buttons[fakerc::Mode]))
      {
        in_mode_ = !in_mode_;
        std::cout << "[InMode] " << in_mode_ << std::endl;
      }

      if (fabs(fakerc_cmd.buttons[fakerc::Gear]))
      {
        in_gear_ = !in_gear_;
        std::cout << "[InGear] " << in_gear_ << std::endl;
      }
    }

    void updateModeChannel()
    {
        if (fakerc_cmd.axes[fakerc::Mode] < -0.5)
        {
            fakerc_cmd.axes[fakerc::Mode] = 0.0;
        }
        else if (fakerc_cmd.axes[fakerc::Mode] > -0.5)
        {
            fakerc_cmd.axes[fakerc::Mode] = -1.0;
        }
    } 


    void ch5CallBack(const geometry_msgs::PoseWithCovarianceStampedConstPtr& msg)
    {
      updateModeChannel();
      std::cout << "ch5CallBack" << std::endl;
      std::cout << "Mode:" << fakerc_cmd.axes[fakerc::Mode] << std::endl;
      return;
    }

    void updateGearChannel()
    {
      if (fakerc_cmd.axes[fakerc::Gear] < -0.5)
      {
        fakerc_cmd.axes[fakerc::Gear] = 1.0;
      }
      else if (fakerc_cmd.axes[fakerc::Gear] > -0.5)
      {
        fakerc_cmd.axes[fakerc::Gear] = -1.0;
      }
    } 


    void ch6CallBack(const geometry_msgs::PointStampedConstPtr& msg)
    {
      updateGearChannel();
      std::cout << "ch6CallBack" << std::endl;
      std::cout << "Gear:" << fakerc_cmd.axes[fakerc::Gear] << std::endl;
      return;
    }

    bool fakercAvailable() 
    {
      // if ((ros::Time::now() - last_msg_time) > ros::Duration(joystick_timeout_)) 
      // {
      //   return false;
      // }

      return true;
    }

    void mainLoop(const ros::WallTimerEvent& time) 
    {
      mavros_msgs::RCIn rc_msg;
      rc_msg.header.frame_id = "world";
      rc_msg.header.stamp = ros::Time::now();
      for (size_t i=0; i<4; i++)
        rc_msg.channels.push_back(1500);
      for (size_t i=4; i<8; i++)
        rc_msg.channels.push_back(1000);

      if (fakercAvailable()) 
      {
        rc_msg.channels[4] = (uint)(1000.0 * in_mode_ + 1000);
        rc_msg.channels[5] = (uint)(1000.0 * in_gear_ + 1000);

        if (fabs(fakerc_cmd.axes[fakerc::W1]) > zero_tolerance)
        {
          rc_msg.channels[fakerc::W1] = (uint)(fakerc_cmd.axes[fakerc::W1] * 500.0 + 1500);
        }

        if (fabs(fakerc_cmd.axes[fakerc::W2]) > zero_tolerance) 
        {
          rc_msg.channels[fakerc::W2] = (uint)(fakerc_cmd.axes[fakerc::W2] * 500.0 + 1500);
        }

        if (fabs(fakerc_cmd.axes[fakerc::W3]) > zero_tolerance) 
        {
          rc_msg.channels[fakerc::W3] = (uint)(fakerc_cmd.axes[fakerc::W3] * 500.0 + 1500);
        }
      }
      fakerc_pub.publish(rc_msg);
    }

    void rvizLoop(const ros::WallTimerEvent& time)
    {
      mavros_msgs::RCIn rc_msg;
      rc_msg.header.frame_id = "world";
      rc_msg.header.stamp = ros::Time::now();
      for (size_t i=0; i<4; i++)
        rc_msg.channels.push_back(1500);
      for (size_t i=4; i<8; i++)
        rc_msg.channels.push_back(1000);

      rc_msg.channels[4] = (uint)(-1000.0 * fakerc_cmd.axes[fakerc::Mode] + 1000);
      rc_msg.channels[5] = (uint)(-1000.0 * fakerc_cmd.axes[fakerc::Gear] + 1000);

      if (fabs(fakerc_cmd.axes[fakerc::W1]) > zero_tolerance)
      {
        rc_msg.channels[fakerc::W1] = (uint)(fakerc_cmd.axes[fakerc::W1] * 500.0 + 1500);
      }

      if (fabs(fakerc_cmd.axes[fakerc::W2]) > zero_tolerance) 
      {
        rc_msg.channels[fakerc::W2] = (uint)(fakerc_cmd.axes[fakerc::W2] * 500.0 + 1500);
      }

      if (fabs(fakerc_cmd.axes[fakerc::W3]) > zero_tolerance) 
      {
        rc_msg.channels[fakerc::W3] = (uint)(fakerc_cmd.axes[fakerc::W3] * 500.0 + 1500);
      }

      fakerc_pub.publish(rc_msg);
    }
};

int main(int argc, char** argv) 
{
  ros::init(argc, argv, "fake_rc_node");
  
  FakeRC fake_rc;
  
  ros::spin();
  
  return 0;
}
