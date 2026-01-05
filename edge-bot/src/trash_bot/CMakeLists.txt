cmake_minimum_required(VERSION 3.8)
project(trash_bot)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(irobot_create_msgs REQUIRED)

# Generate custom messages and services
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BinDetection.msg"
  "msg/RobotCommand.msg"
  "srv/ClassifyBin.srv"
  "srv/ExecuteCommand.srv"
  "srv/FindBin.srv"
  DEPENDENCIES builtin_interfaces std_msgs
)

# C++ executables
add_executable(motion_subscriber src/motion_subscriber.cpp)
ament_target_dependencies(motion_subscriber
  rclcpp
  rclcpp_action
  std_msgs
  geometry_msgs
  irobot_create_msgs
)

add_executable(bin_detector_node src/bin_detector_node.cpp)
ament_target_dependencies(bin_detector_node
  rclcpp
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
)

# Link custom messages to C++ executables
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(motion_subscriber "${cpp_typesupport_target}")
target_link_libraries(bin_detector_node "${cpp_typesupport_target}")

# Install C++ executables
install(TARGETS
  motion_subscriber
  bin_detector_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python scripts as executables
install(PROGRAMS
  trash_bot/command_service.py
  trash_bot/command_client.py
  trash_bot/classifier_node.py
  trash_bot/navigation_vlm_node.py
  trash_bot/smolvlm_navigator.py
  trash_bot/camera_publisher_node.py
  trash_bot/map_publisher_node.py
  trash_bot/classification_camera_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python module files (non-executable) that can be imported by the scripts
install(FILES
  trash_bot/llama_vision_client.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files and config
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
