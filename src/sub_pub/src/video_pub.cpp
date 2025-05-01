#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "opencv2/opencv.hpp"
#include "cv_bridge/cv_bridge.h"


using Image = sensor_msgs::msg::Image;
class VideoPublisher : public rclcpp::Node
{
private:
    rclcpp::Publisher<Image>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    cv::VideoCapture cap_;
    double fps_;
    void publish_video();
    int read_video(const std::string& video_path)
    {
        // 打开视频文件
        cap_.open(video_path, cv::CAP_V4L2);
        fps_ = cap_.get(cv::CAP_PROP_FPS);
        return cap_.isOpened() ? 1 : 0;
    }

public:
    VideoPublisher(): Node("video_publisher")
    {
        publisher_ = this->create_publisher<Image>("video_topic", 10);
        // 读取视频文件
        auto ret = read_video("/home/arcray/ProgramFiles/ROS2_Learning/Xbot/src/sub_pub/videos/1080P.mp4");
        if (ret == 0)
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to read video file");
            return;
        }
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int64_t>(1000/fps_)), std::bind(&VideoPublisher::publish_video, this));
    }
};

void VideoPublisher::publish_video() {
  cv::Mat frame;
  cap_.read(frame);
  // 转换为ROS2的Image消息
  auto msg =
      cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
  msg->header.stamp = this->get_clock()->now();
  auto frameNumber = static_cast<int>(cap_.get(cv::CAP_PROP_POS_FRAMES));
  // 发布消息
  publisher_->publish(*msg);
  RCLCPP_INFO(this->get_logger(), "FPS: %.1f,Video frame published: %d", fps_, frameNumber);
}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VideoPublisher>());
    rclcpp::shutdown();
    return 0;
}