/**
 * @file camera_sub.hpp
 * @brief CameraSubscriber class head file
 * @date 2025-05-22
 * @author ArcRay
 */

#ifndef CAMERA_SUB_HPP
#define CAMERA_SUB_HPP

#include "rclcpp/rclcpp.hpp"
#include "interfaces/msg/frame_info.hpp"
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <chrono>
#include <functional>

using FrameInfo = interfaces::msg::FrameInfo;

class CameraSubscriber : public rclcpp::Node
{
private:
    rclcpp::Subscription<FrameInfo>::SharedPtr subscription_;
    std::chrono::high_resolution_clock::time_point startTime_;

    void ImageCallback(const FrameInfo::SharedPtr msg);
    void ImageDisplay(const cv::Mat& image);

public:
    CameraSubscriber();
    ~CameraSubscriber() = default;
};

#endif // CAMERA_SUB_HPP