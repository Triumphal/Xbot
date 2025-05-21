/**
 * @file camera_sub.cpp
 * @brief CameraSubscriber class implementation
 * @date 2025-05-22
 * @author ArcRay
 */
#include "camera/camera_sub.hpp"

#include "sensor_msgs/msg/image.hpp"
using Image = sensor_msgs::msg::Image;

CameraSubscriber::CameraSubscriber() : Node("camera_subscriber") {
  subscription_ = this->create_subscription<FrameInfo>(
      "camera/frame_info", 10,
      std::bind(&CameraSubscriber::ImageCallback, this, std::placeholders::_1));
}

void CameraSubscriber::ImageCallback(const FrameInfo::SharedPtr msg) {
  try {
    // 将压缩的图像数据转换为 OpenCV 格式
    cv::Mat image = cv::imdecode(cv::Mat(msg->image.data), cv::IMREAD_COLOR);
    ImageDisplay(image);
  } catch (std::exception &e) {
    RCLCPP_ERROR(this->get_logger(), "Exception: %s", e.what());
  }
}

void CameraSubscriber::ImageDisplay(const cv::Mat &image) {
  auto now = std::chrono::high_resolution_clock::now();
  auto timeDelta =
      std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime_)
          .count();
  startTime_ = now;
  std::ostringstream fps_stream;
  fps_stream << std::fixed << std::setprecision(2) << 1000.0 / timeDelta;

  std::string fps = "FPS: " + fps_stream.str();
  std::string imageInfo =
      std::to_string(image.cols) + "x" + std::to_string(image.rows);

  cv::putText(image, fps + " " + imageInfo, cv::Point(50, 50),
              cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 0, 0), 2);
  cv::imshow("Camera Video Stream", image);
  cv::waitKey(1);
}

int main(int argc, char *argv[]) {
  // 初始化 ROS2
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CameraSubscriber>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}