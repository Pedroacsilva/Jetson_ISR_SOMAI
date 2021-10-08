// Code from
// http://wiki.ros.org/cv_bridge/Tutorials/UsingCvBridgeToConvertBetweenROSImagesAndOpenCVImages#cv_bridge.2FTutorials.2FUsingCvBridgeCppHydro.Converting_ROS_image_messages_to_OpenCV_images
// Change as needed

#include "opencv2/opencv.hpp"
//#include "pcl_ros/point_cloud.h"
#include <DarkHelp.hpp>
#include <cmath>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>

static const std::string OPENCV_ZED_FEED = "ZED Camera Feed (OpenCV)";
static const std::string OPENCV_FACEMASKS = "Face Masks Detections";
static const std::string OPENCV_DISTANCES = "Social Distancing";
static const std::string face_config_file =
    "/home/jetsonisr/src_nn/face_mask_nn/face_mask.cfg";
static const std::string face_weights_file =
    "/home/jetsonisr/src_nn/face_mask_nn/face_mask_best.weights";
static const std::string face_names_file =
    "/home/jetsonisr/src_nn/face_mask_nn/face_mask.names";

struct point3d {
  float x, y, z;
};

struct img_point2d {
  int u, v;
};

// typedef sensor_msgs::Image::ConstPtr depth_image;
// typedef cv_bridge::CvImage cv_image;
typedef cv_bridge::CvImagePtr cv_image_ptr;

class RGB_Depth_Subscriber {

  ros::NodeHandle _n_rgb_depth;
  image_transport::ImageTransport _it;
  image_transport::Subscriber _image_sub;
  ros::Subscriber _depth_sub;
  DarkHelp::PredictionResults face_mask_results;
  cv::Mat RGB_Image;
  float *Depth_Image;
  std::vector<point3d> detection_world_coords;
  std::vector<cv::Point2i> image_coords;

public:
  RGB_Depth_Subscriber() : _it(_n_rgb_depth) {
    // Subsribe to input video feed
    _image_sub = _it.subscribe("/zed2/zed_node/left/image_rect_color", 1,
                               &RGB_Depth_Subscriber::ros2cv, this);
    // Subscribe to depth image
    _depth_sub =
        _n_rgb_depth.subscribe("/zed2/zed_node/point_cloud/cloud_registered",
                               10, &RGB_Depth_Subscriber::depthCallback, this);
  }

  ~RGB_Depth_Subscriber() {
    // cv::destroyWindow(OPENCV_ZED_FEED);
    cv::destroyWindow(OPENCV_FACEMASKS);
  }
  void process_detections(DarkHelp::PredictionResults &face_mask_results,
                          std::vector<point3d> &detection_world_coords,
                          const sensor_msgs::PointCloud2Ptr &pCloud) noexcept {
    int i = 0;
    image_coords.clear();
    for (auto detection : face_mask_results) {
      cv::Point2i detection_center = {
          detection.rect.x + detection.rect.width / 2,
          detection.rect.y + detection.rect.height / 2};
      image_coords.emplace_back(detection_center);
      int array_pos = detection_center.y * pCloud->row_step +
                      detection_center.x * pCloud->point_step;
      int array_pos_x = array_pos + pCloud->fields[0].offset;
      int array_pos_y = array_pos + pCloud->fields[1].offset;
      int array_pos_z = array_pos + pCloud->fields[2].offset;
      point3d world_coords;
      memcpy(&world_coords.x, &pCloud->data[array_pos_x], sizeof(float));
      memcpy(&world_coords.y, &pCloud->data[array_pos_y], sizeof(float));
      memcpy(&world_coords.z, &pCloud->data[array_pos_z], sizeof(float));
      detection_world_coords.emplace_back(world_coords);
    }
  }

  // void depthCallback(const sensor_msgs::Image::ConstPtr& msg) {
  // void depthCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
  // void depthCallback(const pcl::PCLPointCloud2ConstPtr& pCloud) {
  void depthCallback(const sensor_msgs::PointCloud2Ptr &pCloud) noexcept {

    // Reset detections
    detection_world_coords.clear();

    process_detections(face_mask_results, detection_world_coords, pCloud);
    int i = 0;
    int j = 1;
    float distance = 0.0;
    std::for_each(
        detection_world_coords.begin(), detection_world_coords.end(),
        [&](const point3d &coords1) {
          std::for_each(
              detection_world_coords.begin() + i + 1,
              detection_world_coords.end(), [&](const point3d &coords2) {
                distance = std::sqrt(std::pow(coords2.x - coords1.x, 2) +
                                     std::pow(coords2.y - coords1.y, 2) +
                                     std::pow(coords2.z - coords1.z, 2));
                distance < 2.0
                    ? cv::line(RGB_Image, image_coords[i], image_coords[j],
                               cv::Scalar(0, 0, 255), 8, cv::LINE_AA)
                    : cv::line(RGB_Image, image_coords[i], image_coords[j],
                               cv::Scalar(0, 255, 0), 8, cv::LINE_AA);
                j++;
              });
          i++;
          j = 1;
        });
    cv::namedWindow(OPENCV_DISTANCES, cv::WINDOW_AUTOSIZE);
    cv::imshow(OPENCV_DISTANCES, RGB_Image);
    cv::waitKey(3);
  }


  void ros2cv(const sensor_msgs::Image::ConstPtr &msg) {
    static cv_image_ptr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception &e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    RGB_Image = cv_ptr->image;
    static DarkHelp nn_face_mask(face_config_file, face_weights_file, face_names_file);
    nn_face_mask.threshold = 0.35;
    nn_face_mask.include_all_names = false;
    nn_face_mask.names_include_percentage = true;
    nn_face_mask.annotation_include_duration = true;
    nn_face_mask.annotation_include_timestamp = false;
    nn_face_mask.sort_predictions = DarkHelp::ESort::kAscending;

    face_mask_results = nn_face_mask.predict(cv_ptr->image);
    cv::Mat face_mask_output = nn_face_mask.annotate();
    cv::namedWindow(OPENCV_FACEMASKS, cv::WINDOW_AUTOSIZE);
    cv::imshow(OPENCV_FACEMASKS, face_mask_output);
    cv::waitKey(3);
  }
};

  int main(int argc, char **argv) {
    ros::init(argc, argv, "RGB_Depth_DL");
    RGB_Depth_Subscriber rgbd_s;
    ros::spin();
  }
