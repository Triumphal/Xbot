#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include "cv_bridge/cv_bridge.h"
#include <chrono>
#include <functional>

using Image = sensor_msgs::msg::Image;

class CameraSubscriber : public rclcpp::Node
{
private:
    rclcpp::Subscription<Image>::SharedPtr subscription_;
    std::chrono::high_resolution_clock::time_point startTime_;

    void ImageCallback(const Image::SharedPtr msg);
    void ImageDisplay(const cv::Mat& image);

public:
    CameraSubscriber(): Node("camera_subscriber")
    {
        subscription_ = this->create_subscription<Image>("image_raw", 10, std::bind(&CameraSubscriber::ImageCallback, this, std::placeholders::_1));
    }
};

void CameraSubscriber::ImageCallback(const Image::SharedPtr msg)
{
    try {
        // 将 ROS Image 转换为 OpenCV Mat
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
        cv::Mat image = cv_ptr->image;
        ImageDisplay(image);
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
}
void CameraSubscriber::ImageDisplay(const cv::Mat& image)
{
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


int main(int argc, char *argv[])
{
    // 初始化 ROS2
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CameraSubscriber>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}