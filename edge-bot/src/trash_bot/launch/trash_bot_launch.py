#!/usr/bin/env python3
"""
trash_bot_launch.py
Launch file for the Trash Bot autonomous bin detection system.

Launches:
- motion_subscriber (C++): Controls TurtleBot4 movement
- bin_detector_node (C++): Detects potential bins from camera
- command_service (Python): High-level command service
- classifier_node (Python): Florence-2 bin classification
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare launch arguments
    camera_device_arg = DeclareLaunchArgument(
        'camera_device',
        default_value='/dev/video0',
        description='USB webcam device for classification'
    )

    server_url_arg = DeclareLaunchArgument(
        'server_url',
        default_value='http://192.168.0.81:5000',
        description='Dashboard server URL'
    )

    max_linear_speed_arg = DeclareLaunchArgument(
        'max_linear_speed',
        default_value='0.3',
        description='Maximum linear speed (m/s)'
    )

    max_angular_speed_arg = DeclareLaunchArgument(
        'max_angular_speed',
        default_value='1.0',
        description='Maximum angular speed (rad/s)'
    )

    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic',
        default_value='/camera/image_raw',
        description='Camera image topic for bin detection'
    )

    oakd_camera_topic_arg = DeclareLaunchArgument(
        'oakd_camera_topic',
        default_value='/oakd/rgb/preview/image_raw',
        description='OAK-D camera topic for SmolVLM navigation'
    )

    llama_server_url_arg = DeclareLaunchArgument(
        'llama_server_url',
        default_value='http://192.168.0.81:5000',
        description='llama.cpp server URL for SmolVLM2 inference'
    )

    enable_navigation_vlm_arg = DeclareLaunchArgument(
        'enable_navigation_vlm',
        default_value='true',
        description='Enable SmolVLM navigation node'
    )

    enable_dashboard_nodes_arg = DeclareLaunchArgument(
        'enable_dashboard_nodes',
        default_value='true',
        description='Enable camera and map publisher nodes for dashboard'
    )

    use_ros_camera_arg = DeclareLaunchArgument(
        'use_ros_camera',
        default_value='false',
        description='Use ROS camera topic (True) or USB camera (False) for navigation'
    )

    use_florence2_arg = DeclareLaunchArgument(
        'use_florence2',
        default_value='true',
        description='Use Florence-2 for object detection + SmolVLM for direction'
    )

    navigation_camera_device_arg = DeclareLaunchArgument(
        'navigation_camera_device',
        default_value='/dev/nav_camera',
        description='USB camera device for navigation (persistent symlink based on serial)'
    )

    classification_camera_device_arg = DeclareLaunchArgument(
        'classification_camera_device',
        default_value='/dev/class_camera',
        description='USB camera device for classification (persistent symlink based on serial)'
    )

    map_topic_arg = DeclareLaunchArgument(
        'map_topic',
        default_value='/map',
        description='SLAM map topic'
    )

    enable_slam_arg = DeclareLaunchArgument(
        'enable_slam',
        default_value='false',  # Disabled by default - TurtleBot4 Lite has no LiDAR
        description='Enable SLAM Toolbox for mapping (requires LiDAR on /scan topic)'
    )

    # Get package share directory
    pkg_share = FindPackageShare('trash_bot')

    # Motion Subscriber Node (C++)
    motion_subscriber_node = Node(
        package='trash_bot',
        executable='motion_subscriber',
        name='motion_subscriber',
        output='screen',
        parameters=[{
            'max_linear_speed': LaunchConfiguration('max_linear_speed'),
            'max_angular_speed': LaunchConfiguration('max_angular_speed'),
        }]
    )

    # Bin Detector Node (C++)
    bin_detector_node = Node(
        package='trash_bot',
        executable='bin_detector_node',
        name='bin_detector_node',
        output='screen',
        parameters=[{
            'camera_topic': LaunchConfiguration('camera_topic'),
            'min_contour_area': 5000,
            'detection_interval': 5,
        }]
    )

    # Command Service Node (Python)
    command_service_node = Node(
        package='trash_bot',
        executable='command_service.py',
        name='command_service',
        output='screen',
    )

    # Classifier Node (Python) - SmolVLM2 via llama.cpp
    classifier_node = Node(
        package='trash_bot',
        executable='classifier_node.py',
        name='classifier_node',
        output='screen',
        parameters=[{
            'llama_server_url': LaunchConfiguration('llama_server_url'),
            'camera_device': LaunchConfiguration('classification_camera_device'),
            'camera_width': 640,
            'camera_height': 480,
            'server_url': LaunchConfiguration('server_url'),
        }]
    )

    # Navigation VLM Node (Python) - Florence-2 + SmolVLM2 via server
    navigation_vlm_node = Node(
        package='trash_bot',
        executable='navigation_vlm_node.py',
        name='navigation_vlm_node',
        output='screen',
        parameters=[{
            'llama_server_url': LaunchConfiguration('llama_server_url'),
            'camera_device': LaunchConfiguration('navigation_camera_device'),
            'use_ros_camera': LaunchConfiguration('use_ros_camera'),
            'camera_topic': LaunchConfiguration('oakd_camera_topic'),
            'use_florence2': LaunchConfiguration('use_florence2'),
            'inference_rate': 0.5,  # Run inference every 2 seconds (fast with server GPU)
            'forward_speed': 1.0,  # Speed factor 0-1 (1.0 = full max_linear_speed = 0.3 m/s)
            'turn_speed': 0.5,  # Speed factor 0-1 (0.5 = 50% of max_angular_speed = 0.5 rad/s)
            'search_timeout': 300.0,  # 5 minutes
        }]
    )

    # Camera Publisher Node (Python) - Streams navigation camera to dashboard
    camera_publisher_node = Node(
        package='trash_bot',
        executable='camera_publisher_node.py',
        name='camera_publisher_node',
        output='screen',
        parameters=[{
            'server_url': LaunchConfiguration('server_url'),
            'camera_device': LaunchConfiguration('navigation_camera_device'),  # Use navigation camera for main dashboard feed
            'publish_rate': 0.5,  # Every 2 seconds
            'username': 'admin',
            'password': 'admin',
        }]
    )

    # Map Publisher Node (Python) - Streams SLAM map to dashboard
    map_publisher_node = Node(
        package='trash_bot',
        executable='map_publisher_node.py',
        name='map_publisher_node',
        output='screen',
        parameters=[{
            'server_url': LaunchConfiguration('server_url'),
            'map_topic': LaunchConfiguration('map_topic'),
            'publish_rate': 0.2,  # Every 5 seconds
            'username': 'admin',
            'password': 'admin',
        }]
    )

    # SLAM Toolbox Node - Real-time SLAM mapping
    slam_params_file = PathJoinSubstitution([
        pkg_share, 'config', 'slam_params.yaml'
    ])

    slam_toolbox_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[slam_params_file],
        condition=IfCondition(LaunchConfiguration('enable_slam')),
    )

    # Classification Camera Publisher Node - Streams classification camera to dashboard
    classification_camera_publisher_node = Node(
        package='trash_bot',
        executable='classification_camera_publisher.py',
        name='classification_camera_publisher',
        output='screen',
        parameters=[{
            'server_url': LaunchConfiguration('server_url'),
            'camera_device': LaunchConfiguration('classification_camera_device'),
            'publish_rate': 0.5,  # Every 2 seconds
            'username': 'admin',
            'password': 'admin',
        }]
    )

    return LaunchDescription([
        # Launch arguments
        camera_device_arg,
        server_url_arg,
        max_linear_speed_arg,
        max_angular_speed_arg,
        camera_topic_arg,
        oakd_camera_topic_arg,
        llama_server_url_arg,
        enable_navigation_vlm_arg,
        enable_dashboard_nodes_arg,
        use_ros_camera_arg,
        use_florence2_arg,
        navigation_camera_device_arg,
        classification_camera_device_arg,
        map_topic_arg,
        enable_slam_arg,

        # Log startup info
        LogInfo(msg='Starting Trash Bot Autonomous System (Florence-2 + SmolVLM2)'),
        LogInfo(msg=['Camera device: ', LaunchConfiguration('camera_device')]),
        LogInfo(msg=['Server URL: ', LaunchConfiguration('server_url')]),
        LogInfo(msg=['Vision server: ', LaunchConfiguration('llama_server_url')]),
        LogInfo(msg=['Florence-2 enabled: ', LaunchConfiguration('use_florence2')]),

        # Nodes
        motion_subscriber_node,
        bin_detector_node,
        command_service_node,
        classifier_node,
        navigation_vlm_node,

        # Dashboard streaming nodes
        # NOTE: camera_publisher_node disabled - navigation_vlm_node now posts frames to dashboard
        # This prevents camera resource contention (/dev/nav_camera can only be opened by one node)
        # camera_publisher_node,
        map_publisher_node,

        # SLAM mapping
        slam_toolbox_node,

        # Classification camera streaming (second camera feed)
        classification_camera_publisher_node,
    ])
