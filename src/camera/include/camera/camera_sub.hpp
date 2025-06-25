/**
 * @file camera_sub.hpp
 * @brief CameraSubscriber class head file
 * @date 2025-05-22
 * @author ArcRay
 */

#ifndef CAMERA_SUB_HPP
#define CAMERA_SUB_HPP

#include <QApplication>
#include <QLabel>
#include <QPixmap>
#include <chrono>
#include <functional>
#include <opencv2/highgui/highgui.hpp>

#include "interfaces/msg/frame_info.hpp"
#include "opencv2/opencv.hpp"
#include "rclcpp/rclcpp.hpp"

using FrameInfo = interfaces::msg::FrameInfo;

class CameraSubscriber : public rclcpp::Node {
 private:
  QLabel* label_ = new QLabel();
  rclcpp::Subscription<FrameInfo>::SharedPtr subscription_;
  std::chrono::high_resolution_clock::time_point startTime_;

  void ImageCallback(const FrameInfo::SharedPtr msg);
  void ImageDisplay(const cv::Mat& image);
  void AddInformation(cv::Mat& image);
  void ImageDisplayWithQt(const cv::Mat& image);

 public:
  CameraSubscriber(const std::string& tipic_name);
  ~CameraSubscriber() = default;
};

#endif  // CAMERA_SUB_HPP