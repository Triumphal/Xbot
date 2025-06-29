import launch
import os
from ament_index_python.packages import get_package_share_directory
from ament_index_python.search_paths import get_search_paths
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration,PathJoinSubstitution


def generate_launch_description():
    # 通过外部参数传参数
    action_camera_sub_para = DeclareLaunchArgument("sub_topic",default_value="detector/detect")

    # 通过配置文件传默认默认参数
    detect_python_config_file = os.path.join(
        get_package_share_directory("detect_python"), 
        'config', 
        'params.yaml')

    # 相机发布节点
    action_node_camera_pub = Node(
        package="camera",
        executable="camera_pub",
        output="log",
        arguments=["--ros-args", "--log-level", "WARN"],
    )
    # 订阅节点 显示结果
    action_node_camera_sub = Node(
        package="camera",
        executable="camera_sub",
        output="log",
        parameters=[{"sub_topic": LaunchConfiguration("sub_topic")}], # 
        arguments=["--ros-args", "--log-level", "WARN"],
    )
    action_node_detect = Node(
        package="detect_python",
        executable="detect_sub_pub",
        output="screen",
        name="detect_sub_pub",
        parameters=[detect_python_config_file],
        arguments=["--ros-args", "--log-level", "WARN"],
    )
    return launch.LaunchDescription(
        [
            # action动作
            action_camera_sub_para,
            action_node_camera_pub,
            action_node_detect,
            action_node_camera_sub,
        ]
    )
