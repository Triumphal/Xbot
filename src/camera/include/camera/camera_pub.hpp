/**
 * @file camera_pub.hpp
 * @brief CameraPublisher class head file
 * @date 2025-05-22
 * @author ArcRay
 */
#ifndef CAMERA_PUB_HPP
#define CAMERA_PUB_HPP

#include <chrono>

#include "interfaces/msg/frame_info.hpp"
#include "opencv2/opencv.hpp"
#include "rclcpp/rclcpp.hpp"

using FrameInfo = interfaces::msg::FrameInfo;
using namespace std::chrono_literals;

class CameraPublisher : public rclcpp::Node {
 private:
  rclcpp::Publisher<FrameInfo>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  cv::VideoCapture cap_;
  uint64 frame_id_ = 0;

  void timer_callback();

 public:
  CameraPublisher(const int camera_num);
  ~CameraPublisher();
};

#endif  // CAMERA_PUB_HPP