cmake_minimum_required(VERSION 3.8)
project(sub_pub)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)


add_executable(image_pub src/image_pub.cpp)
add_executable(video_pub src/video_pub.cpp)
add_executable(camera_pub src/camera_pub.cpp)
add_executable(camera_sub src/camera_sub.cpp)

ament_target_dependencies(image_pub
  rclcpp
  sensor_msgs
  std_msgs
  builtin_interfaces
  OpenCV
  cv_bridge
)

ament_target_dependencies(video_pub
  rclcpp
  sensor_msgs
  std_msgs
  builtin_interfaces
  OpenCV
  cv_bridge
)

ament_target_dependencies(camera_pub
  rclcpp
  sensor_msgs
  std_msgs
  builtin_interfaces
  OpenCV
  cv_bridge
)

ament_target_dependencies(camera_sub
  rclcpp
  sensor_msgs
  std_msgs
  builtin_interfaces
  OpenCV
  cv_bridge
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(TARGETS image_pub video_pub camera_pub camera_sub
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
