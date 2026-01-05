#!/usr/bin/env python3
"""
retrace.py - Retrace robot path using command history with proper action waiting.

This script reads command history and executes reverse commands, properly waiting
for each action to complete before sending the next command.
"""

import sys
import os
import json
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from irobot_create_msgs.action import DriveDistance, RotateAngle

HISTORY_FILE = os.path.expanduser("~/ROS/logs/command_history.json")
MAX_LINEAR_SPEED = 0.15  # m/s - matches motion_subscriber
MAX_ANGULAR_SPEED = 1.0  # rad/s

# ANSI colors
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
CYAN = '\033[0;36m'
NC = '\033[0m'


class RetraceNode(Node):
    """Node to retrace robot path by reversing command history."""

    def __init__(self):
        super().__init__('retrace_node')

        self.callback_group = ReentrantCallbackGroup()

        # Action clients
        self.drive_client = ActionClient(
            self, DriveDistance, '/_do_not_use/drive_distance',
            callback_group=self.callback_group
        )
        self.rotate_client = ActionClient(
            self, RotateAngle, '/_do_not_use/rotate_angle',
            callback_group=self.callback_group
        )

        self.get_logger().info('Retrace Node initialized')

    def wait_for_servers(self, timeout=10.0):
        """Wait for action servers to be available."""
        self.get_logger().info('Waiting for action servers...')

        if not self.drive_client.wait_for_server(timeout_sec=timeout):
            self.get_logger().error('Drive action server not available!')
            return False

        if not self.rotate_client.wait_for_server(timeout_sec=timeout):
            self.get_logger().error('Rotate action server not available!')
            return False

        self.get_logger().info('Action servers ready!')
        return True

    def execute_drive(self, distance, max_speed, timeout=30.0):
        """Execute drive action and wait for completion.

        For backward motion (negative distance), we use rotate-forward-rotate
        because Create3's DriveDistance doesn't reliably complete with negative distances.
        """
        if distance < 0:
            # BACKWARD: Create3 DriveDistance doesn't complete reliably with negative distance
            # Use: rotate 180° -> drive forward -> rotate 180° back
            self.get_logger().info(f'BACKWARD {abs(distance):.2f}m - using rotate-forward-rotate method')

            # Rotate 180 degrees (π radians)
            if not self.execute_rotate(3.14159, 0.5, timeout=15.0):
                self.get_logger().warn('Failed to rotate 180 for backward')
                return False
            time.sleep(0.5)

            # Drive forward the absolute distance
            if not self._do_forward_drive(abs(distance), max_speed, timeout):
                self.get_logger().warn('Failed forward drive during backward maneuver')
                return False
            time.sleep(0.5)

            # Rotate back 180 degrees
            if not self.execute_rotate(3.14159, 0.5, timeout=15.0):
                self.get_logger().warn('Failed to rotate back 180')
                return False

            return True
        else:
            return self._do_forward_drive(distance, max_speed, timeout)

    def _do_forward_drive(self, distance, max_speed, timeout=30.0):
        """Execute forward drive action and wait for completion."""
        goal = DriveDistance.Goal()
        goal.distance = distance
        goal.max_translation_speed = max_speed

        self.get_logger().info(f'Driving FORWARD {distance:.2f}m at max {max_speed:.2f} m/s')

        # Send goal
        send_goal_future = self.drive_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=5.0)

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Drive goal rejected!')
            return False

        self.get_logger().info('Drive goal accepted, waiting for completion...')

        # Wait for result
        result_future = goal_handle.get_result_async()

        # Spin until complete or timeout
        start_time = time.time()
        while not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.5)
            elapsed = time.time() - start_time
            if elapsed > timeout:
                self.get_logger().warn(f'Drive timeout after {elapsed:.1f}s - canceling')
                goal_handle.cancel_goal_async()
                return False
            if elapsed % 5 < 0.5:  # Log every ~5 seconds
                self.get_logger().info(f'  Still driving... {elapsed:.1f}s elapsed')

        result = result_future.result()
        if result.status == 4:  # SUCCEEDED
            self.get_logger().info(f'Drive completed successfully!')
            return True
        elif result.status == 5:  # CANCELED
            self.get_logger().warn('Drive was canceled')
            return False
        else:
            self.get_logger().warn(f'Drive ended with status {result.status}')
            return False

    def execute_rotate(self, angle, max_speed, timeout=30.0):
        """Execute rotate action and wait for completion."""
        goal = RotateAngle.Goal()
        goal.angle = angle
        goal.max_rotation_speed = max_speed

        direction = "LEFT" if angle > 0 else "RIGHT"
        self.get_logger().info(f'Rotating {direction} {abs(angle):.2f}rad at max {max_speed:.2f} rad/s')

        # Send goal
        send_goal_future = self.rotate_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=5.0)

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Rotate goal rejected!')
            return False

        self.get_logger().info('Rotate goal accepted, waiting for completion...')

        # Wait for result
        result_future = goal_handle.get_result_async()

        start_time = time.time()
        while not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.5)
            elapsed = time.time() - start_time
            if elapsed > timeout:
                self.get_logger().warn(f'Rotate timeout after {elapsed:.1f}s - canceling')
                goal_handle.cancel_goal_async()
                return False

        result = result_future.result()
        if result.status == 4:  # SUCCEEDED
            self.get_logger().info(f'Rotate completed successfully!')
            return True
        elif result.status == 5:  # CANCELED
            self.get_logger().warn('Rotate was canceled')
            return False
        else:
            self.get_logger().warn(f'Rotate ended with status {result.status}')
            return False

    def retrace(self, commands):
        """Execute reverse commands to retrace path."""
        total = len(commands)
        self.get_logger().info(f'{CYAN}Starting retrace of {total} commands...{NC}')

        # Process in reverse order
        for i, cmd in enumerate(reversed(commands)):
            cmd_type = cmd['command']
            speed = cmd['speed']
            duration = cmd['duration']

            # Map to reverse command
            reverse_map = {
                'forward': 'backward',
                'backward': 'forward',
                'left': 'right',
                'right': 'left'
            }

            rev_cmd = reverse_map.get(cmd_type)
            if not rev_cmd:
                self.get_logger().warn(f'Unknown command {cmd_type}, skipping')
                continue

            print(f'\n{CYAN}[{i+1}/{total}] {rev_cmd.upper()} (was {cmd_type}) speed={speed} duration={duration}{NC}')

            # Calculate parameters
            speed_factor = max(0.0, min(1.0, speed))

            if rev_cmd in ['forward', 'backward']:
                # Drive command
                distance = MAX_LINEAR_SPEED * speed_factor * duration
                if rev_cmd == 'backward':
                    distance = -distance
                max_speed = MAX_LINEAR_SPEED * speed_factor

                success = self.execute_drive(distance, max_speed, timeout=60.0)
            else:
                # Rotate command
                angle = MAX_ANGULAR_SPEED * speed_factor * duration
                if rev_cmd == 'right':
                    angle = -angle
                max_speed = MAX_ANGULAR_SPEED * speed_factor

                success = self.execute_rotate(angle, max_speed, timeout=30.0)

            if not success:
                self.get_logger().warn(f'{YELLOW}Command {i+1} did not complete successfully{NC}')

            # Small pause between commands
            time.sleep(0.5)

        print(f'\n{GREEN}Retrace complete!{NC}')
        return True


def main():
    print(f'{GREEN}=== Retrace Path Script ==={NC}')
    print(f'History file: {CYAN}{HISTORY_FILE}{NC}')
    print()

    # Check history file
    if not os.path.exists(HISTORY_FILE):
        print(f'{YELLOW}No command history found at {HISTORY_FILE}{NC}')
        print('Nothing to retrace.')
        return 0

    # Load history
    try:
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
    except Exception as e:
        print(f'{RED}Failed to load history: {e}{NC}')
        return 1

    commands = history.get('commands', [])
    total = history.get('total_commands', len(commands))

    if not commands:
        print(f'{YELLOW}No commands in history{NC}')
        return 0

    print(f'Found {total} commands to retrace')

    # Initialize ROS
    rclpy.init()

    try:
        node = RetraceNode()

        # Wait for action servers
        if not node.wait_for_servers(timeout=15.0):
            print(f'{RED}Failed to connect to action servers!{NC}')
            return 1

        # Execute retrace
        node.retrace(commands)

    except KeyboardInterrupt:
        print(f'\n{YELLOW}Interrupted by user{NC}')
    finally:
        rclpy.shutdown()

    return 0


if __name__ == '__main__':
    sys.exit(main())
