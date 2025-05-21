/**
 * @file camera_pub.hpp
 * @brief CameraPublisher class head file
 * @date 2025-05-22
 * @author ArcRay
 */
#ifndef CAMERA_PUB_HPP
#define CAMERA_PUB_HPP

#include "interfaces/msg/frame_info.hpp"
#include "rclcpp/rclcpp.hpp"
#include "opencv2/opencv.hpp"
#include "chrono"

using FrameInfo = interfaces::msg::FrameInfo;
using namespace std::chrono_literals;

class CameraPublisher : public rclcpp::Node
{
private:
    rclcpp::Publisher<FrameInfo>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    cv::VideoCapture cap_;
    u_int64_t frame_count_ = 0;
    
    void timer_callback();

public:
    CameraPublisher(const int camera_num);
    ~CameraPublisher();
};

#endif // CAMERA_PUB_HPP