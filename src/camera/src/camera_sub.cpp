/**
 * @file camera_sub.cpp
 * @brief CameraSubscriber class implementation // 订阅相机数据 并显示
 * @date 2025-05-22
 * @author ArcRay
 */

#include "camera/camera_sub.hpp"

CameraSubscriber::CameraSubscriber(const std::string &tipic_name) : Node("camera_subscriber") {
  subscription_ = this->create_subscription<FrameInfo>(
      tipic_name, 10, std::bind(&CameraSubscriber::ImageCallback, this, std::placeholders::_1));
}

void CameraSubscriber::ImageCallback(const FrameInfo::SharedPtr msg) {
  try {
    // 将压缩的图像数据转换为 OpenCV 格式
    cv::Mat image = cv::imdecode(cv::Mat(msg->image.data), cv::IMREAD_COLOR);
    // ImageDisplay(image);
    AddInformation(image);
    ImageDisplayWithQt(image);
  } catch (std::exception &e) {
    RCLCPP_ERROR(this->get_logger(), "Exception: %s", e.what());
  }
}
void CameraSubscriber::AddInformation(cv::Mat &image) {
  // 在图像上添加文本信息
  auto now = std::chrono::high_resolution_clock::now();
  auto timeDelta = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime_).count();
  startTime_ = now;
  std::ostringstream fps_stream;
  fps_stream << std::fixed << std::setprecision(2) << 1000.0 / timeDelta;

  std::string fps = "FPS: " + fps_stream.str();
  std::string imageInfo = std::to_string(image.cols) + "x" + std::to_string(image.rows);

  cv::putText(image, fps + " " + imageInfo, cv::Point(50, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 0, 0), 2);
}

void CameraSubscriber::ImageDisplay(const cv::Mat &image) {
  auto now = std::chrono::high_resolution_clock::now();
  auto timeDelta = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime_).count();
  startTime_ = now;
  std::ostringstream fps_stream;
  fps_stream << std::fixed << std::setprecision(2) << 1000.0 / timeDelta;

  std::string fps = "FPS: " + fps_stream.str();
  std::string imageInfo = std::to_string(image.cols) + "x" + std::to_string(image.rows);

  cv::putText(image, fps + " " + imageInfo, cv::Point(50, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 0, 0), 2);
  cv::imshow("Camera Video Stream", image);
  cv::waitKey(1);
}

void CameraSubscriber::ImageDisplayWithQt(const cv::Mat &image) {
  // 使用 Qt 显示图像
  QPixmap pixmap = QPixmap::fromImage(QImage(image.data, image.cols, image.rows, image.step, QImage::Format_BGR888));
  label_->setPixmap(pixmap.scaled(image.cols, image.rows, Qt::KeepAspectRatio, Qt::SmoothTransformation));
  label_->show();
}

int main(int argc, char *argv[]) {
  // 初始化 ROS2
  rclcpp::init(argc, argv);
  QApplication app(argc, argv);
  auto node = std::make_shared<CameraSubscriber>("detector/detect"); //
  std::thread spin_thread([&]() -> void {
    rclcpp::spin(node);
    rclcpp::shutdown();
  });
  spin_thread.detach();
  app.exec();
  return 0;
}