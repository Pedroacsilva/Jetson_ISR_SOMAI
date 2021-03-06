
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/Joy.h>
#include <sensor_msgs/PointCloud.h>	// sonars
#include <math.h>
#include <stdlib.h>

#define PI 3.141592654

class SafeTeleop
{
public:
  SafeTeleop();
  

private:
  // parameters
  double l_scale, a_scale;	// linear and angular "gains"
  double safe_dist_th, crit_dist_th; // distances used to activate obstacle avoidance
  double hfov;			// half of sonars FoV considered in obstacle avoidance

  // attributes
  ros::NodeHandle nh;

  double vel_inct;
  double oa;	// obstacle avoidance gain
  int state, count;
  ros::Publisher vel_pub;
  ros::Subscriber joy_sub;
  ros::Subscriber sonars_sub;

  // methods
  void joyCallback(const sensor_msgs::Joy::ConstPtr& joy);
  void sonarsCallback(const sensor_msgs::PointCloud::ConstPtr& pt);

  double euclidist(double x, double y) { return sqrt(x*x+y*y); }
  double angle(double x, double y) { return atan2(y, x); }
};

SafeTeleop::SafeTeleop()
{
  ros::NodeHandle pn("~");
  pn.param("sc_angular", a_scale, 0.75);
  pn.param("sc_linear", l_scale, 0.35);
  pn.param("safe_dist_th", safe_dist_th, 0.6);
  pn.param("crit_dist_th", crit_dist_th, 0.35);
  pn.param("hfov", hfov, 10.0);
	
  hfov = hfov * PI / 180.0;
  ROS_INFO("safeteleop node parameters:\n\tl_scale= %.3f  a_scale = %3f  safe_dist_th = %.3f\n \tcrit_dist_th = %.3f  hfov = %.3f", l_scale, a_scale, safe_dist_th, crit_dist_th, hfov);

  oa = 1.0;
  vel_inct = 0.0;
  state = -1;
  count = 0;

  vel_pub = nh.advertise<geometry_msgs::Twist>("cmd_vel", 10);  
  joy_sub = nh.subscribe<sensor_msgs::Joy>("joy", 10, &SafeTeleop::joyCallback, this);
  sonars_sub  = nh.subscribe<sensor_msgs::PointCloud>("sonar", 1, &SafeTeleop::sonarsCallback, this);
}

void SafeTeleop::sonarsCallback(const sensor_msgs::PointCloud::ConstPtr& pt) {
	int size = pt->points.size(); // number of sonars

	double d, a, dmin = 1e99;
	bool eligRead = false;
	int _state = this->state;
	
	for (int i = 0; i < size; i++) {
		d = euclidist(pt->points[i].x, pt->points[i].y);
		a = angle(pt->points[i].x, pt->points[i].y);
		if (a >= -this->hfov && a <= this->hfov && d > 0.0){ // ROSARIA gives zero distance for non-existent sonars
			if (d < dmin) dmin = d;
			eligRead = true;
		}
	}

	if (eligRead){
		if (dmin < this->safe_dist_th){
			if (count > 3) {
				if (dmin > this->crit_dist_th){
					oa = (dmin - this->crit_dist_th) / (this->safe_dist_th - this->crit_dist_th);
					state = 1;
				}
				else { oa = 0.0; state = 2; }
				count = 0;
			}
			count++;
		}
		else { oa = 1.0; state = 0; count = 0;}
	}
	else {
		if (state == -1)
			ROS_WARN("No sonars found within FoV = %.3f deg.", this->hfov*360.0/PI);
		oa = 1.0; state = 0;
	}

	if (state != _state)
		switch (state){
			case 0: ROS_INFO("No obstacles within safe distance of %.4f m", this->safe_dist_th); break;
			case 1: ROS_WARN("Obstacle detected at %.4f m", dmin); break;
			default: ROS_ERROR("Critical obstacle detected at %.4f m", dmin);
		}
}

void SafeTeleop::joyCallback(const sensor_msgs::Joy::ConstPtr& joy)
{
  //ROS_INFO("Callback Wii");
  geometry_msgs::Twist vel;
  static geometry_msgs::Twist vel_; // last velocity
  static bool first_call = true;
  const double delta = 1e-2;
  
  if (first_call) {
  	vel.linear.x = -1.0;	// force to send first command to robot
  	first_call = false;
  }

  if (joy->buttons[4])
	vel_inct = vel_inct + 0.005;

  if (joy->buttons[5])
	vel_inct = vel_inct - 0.005;
  
  if (vel_inct<0)
        vel_inct = 0.0;

  vel.linear.x = (l_scale + vel_inct) * joy->buttons[8]; //pra frente ou zero
  if (vel.linear.x==0.0){
    vel.linear.x = -(l_scale + vel_inct) * joy->buttons[9]; //pra tras ou zero
  }
  
  vel.angular.z = (a_scale + vel_inct) * joy->buttons[6]; //pra esq ou zero
  if (vel.angular.z==0.0){
    vel.angular.z = -(a_scale + vel_inct) * joy->buttons[7]; //pra dta ou zero
  }
  
  // modula velocidade p/ frente e vel. angular usando o ganho de "obstacle avoidance"
  if (vel.linear.x > 0.0) vel.linear.x *= oa;
  vel.angular.z *= oa;

  if (fabs(vel.linear.x - vel_.linear.x) > delta ||
	fabs(vel.angular.z - vel_.angular.z) > delta)// || vel.linear.x == 0.0)
  {
	// publish new command only if the command changes significantly
  	vel_pub.publish(vel);
  	vel_ = vel;
  }

 // ROS_INFO("Publiquei (ANG=%f, LIN=%f)", vel.angular.z, vel.linear.x);  
 //ROS_INFO("Publiquei INC=%f", vel_inct);  
  
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "safeteleop");
  SafeTeleop safeteleop;

  ros::spin();
}
