
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/Joy.h>

class TeleopPioneer
{
public:
  TeleopPioneer();
  

private:
  void joyCallback(const sensor_msgs::Joy::ConstPtr& joy);
  
  ros::NodeHandle nh_;

  double l_scale_, a_scale_;
  double vel_inct;  
  ros::Publisher vel_pub_;
  ros::Subscriber joy_sub_;
  
};

TeleopPioneer::TeleopPioneer():
  a_scale_(1.0),
  l_scale_(0.5),
  vel_inct(0.0)
{

  nh_.param("scale_angular", a_scale_, a_scale_);
  nh_.param("scale_linear", l_scale_, l_scale_);

  vel_pub_ = nh_.advertise<geometry_msgs::Twist>("/ROSARIA/cmd_vel", 1);  
  joy_sub_ = nh_.subscribe<sensor_msgs::Joy>("joy", 10, &TeleopPioneer::joyCallback, this);

}


void TeleopPioneer::joyCallback(const sensor_msgs::Joy::ConstPtr& joy)
{
  //ROS_INFO("Callback Wii\n");
  geometry_msgs::Twist vel;
  static geometry_msgs::Twist vel_; // last velocity
  static bool first_call = true;
  const double delta = 1e-3;
  
  if (first_call) {
  	vel_.linear.x = -1.0;	// force to send first command to robot
  	first_call = false;
  }

  if (joy->buttons[4])
	vel_inct = vel_inct + 0.005;

  if (joy->buttons[5])
	vel_inct = vel_inct - 0.005;
  
  if (vel_inct<0)
        vel_inct = 0.0;

  vel.linear.x = (l_scale_ + vel_inct) * joy->buttons[8]; //pra frente ou zero
  if (vel.linear.x==0.0){
    vel.linear.x = -(l_scale_ + vel_inct) * joy->buttons[9]; //pra tras ou zero
  }
  
  vel.angular.z = (a_scale_ + vel_inct) * joy->buttons[6]; //pra esq ou zero
  if (vel.angular.z==0.0){
    vel.angular.z = -(a_scale_ + vel_inct) * joy->buttons[7]; //pra dta ou zero
  }
  
  	if (fabs(vel.linear.x - vel_.linear.x) > delta ||
			fabs(vel.angular.z - vel_.angular.z) > delta)
	{
		// publish new command only if the command changes significantly
  		vel_pub_.publish(vel);
  		vel_ = vel;
	}

 // ROS_INFO("Publiquei (ANG=%f, LIN=%f)", vel.angular.z, vel.linear.x);  
 //ROS_INFO("Publiquei INC=%f", vel_inct);  
  
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "pioneer_teleop");
  TeleopPioneer teleop_pioneer;

  ros::spin();
}
