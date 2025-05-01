#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "opencv2/opencv.hpp"
#include "cv_bridge/cv_bridge.h"

using Image = sensor_msgs::msg::Image;
class ImagePublisher : public rclcpp::Node
{
public:
    ImagePublisher() : Node("image_publisher")
    {
        publisher_ = this->create_publisher<Image>("image_topic", 10);
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100), std::bind(&ImagePublisher::publish_image, this));
    }

private:
    void publish_image()
    {
        // 读取图片
        cv::Mat image = cv::imread("/home/arcray/ProgramFiles/ROS2_Learning/Xbot/src/sub_pub/images/bus.jpg", cv::IMREAD_COLOR);
        if (image.empty())
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to read image");
            return;
        }

        // 转换为ROS2的Image消息
        auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", image).toImageMsg();
        msg->header.stamp = this->get_clock()->now();

        // 发布消息
        publisher_->publish(*msg);
        RCLCPP_INFO(this->get_logger(), "Image published");
    }

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImagePublisher>());
    rclcpp::shutdown();
    return 0;
}