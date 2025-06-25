/**
 * @file camera_pub.cpp
 * @brief CameraPublisher class implementation // 发布相机数据
 * @date 2025-05-22
 * @author ArcRay
 */
#include "camera/camera_pub.hpp"

CameraPublisher::CameraPublisher(const int camera_num) : Node("camera_publisher") {
  publisher_ = this->create_publisher<FrameInfo>("camera/frame_info", 10);
  cap_.open(camera_num, cv::CAP_V4L2);
  if (!cap_.isOpened()) {
    RCLCPP_ERROR(this->get_logger(), "Don't open camera [%d]", camera_num);
  }
  timer_ = this->create_wall_timer(50ms, std::bind(&CameraPublisher::timer_callback, this));
  // 设置压缩参数
  this->declare_parameter("jpeg_quality", 90);
}

CameraPublisher::~CameraPublisher() {
  if (cap_.isOpened()) {
    cap_.release();
  }
}

void CameraPublisher::timer_callback() {
  cv::Mat frame;
  cap_ >> frame;
  cv::resize(frame, frame, cv::Size(1080, 720));
  if (!frame.empty()) {
    FrameInfo frame_info;
    frame_info.image.header.stamp = this->now();
    frame_info.image.format = "jpeg";
    frame_info.width = frame.cols;
    frame_info.height = frame.rows;
    frame_info.frame_id = frame_id_++;

    // 获取压缩参数质量
    int jpeg_quality = this->get_parameter("jpeg_quality").as_int();
    // 压缩图像
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, jpeg_quality};
    if (cv::imencode(".jpg", frame, frame_info.image.data, params)) {
      publisher_->publish(frame_info);
    } else {
      RCLCPP_ERROR(this->get_logger(), "Failed to encode image");
    }
    if (frame_id_ % 10 == 0) {
      RCLCPP_INFO(this->get_logger(), "Publishing camera frame [%lu]", frame_id_);
    }
  } else {
    RCLCPP_ERROR(this->get_logger(), "Failed to capture frame");
  }
}

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CameraPublisher>(0));
  rclcpp::shutdown();
  return 0;
}