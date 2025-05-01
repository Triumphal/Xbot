#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "opencv2/opencv.hpp"
#include "cv_bridge/cv_bridge.h"
#include "chrono"

using Image = sensor_msgs::msg::Image;
struct FPS
{
    double front_time;
    double last_time;
    double fps;
};

class CameraPublisher : public rclcpp::Node
{
private:
    rclcpp::Publisher<Image>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    cv::VideoCapture cap_;
    FPS fps_;
    u_int64_t frame_count_ = 0;
    void timer_callback();

public:
    CameraPublisher(const int camera_num): Node("camera_publisher")
    {
        publisher_ = this->create_publisher<Image>("image_raw", 10);
        cap_.open(camera_num, cv::CAP_V4L2);
        if (!cap_.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "Don't open camera [%d]", camera_num);
        }
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100), std::bind(&CameraPublisher::timer_callback, this));
    }
};
void CameraPublisher::timer_callback(){
    cv::Mat frame;
    cap_ >> frame;
    cv::resize(frame, frame, cv::Size(1280, 720));
    if (!frame.empty()) {
        auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
        
        publisher_->publish(*msg);
        frame_count_++;
        if (frame_count_ % 10 == 0) {
            RCLCPP_INFO(this->get_logger(), "Publishing camera frame [%lu]", frame_count_);
        }
    } else {
        RCLCPP_ERROR(this->get_logger(), "Failed to capture frame");
    }
}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CameraPublisher>(0));
    rclcpp::shutdown();
    return 0;
}