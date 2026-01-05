#!/usr/bin/env python3
"""
slam_launch.py
Launch file for SLAM Toolbox with TurtleBot4.

Launches slam_toolbox in online async mode for real-time mapping.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    slam_params_file_arg = DeclareLaunchArgument(
        'slam_params_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('trash_bot'),
            'config',
            'slam_params.yaml'
        ]),
        description='Full path to SLAM parameters file'
    )

    # SLAM Toolbox Node
    slam_toolbox_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[
            LaunchConfiguration('slam_params_file'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
    )

    return LaunchDescription([
        # Launch arguments
        use_sim_time_arg,
        slam_params_file_arg,

        # Log startup
        LogInfo(msg='Starting SLAM Toolbox for TurtleBot4'),

        # SLAM node
        slam_toolbox_node,
    ])
