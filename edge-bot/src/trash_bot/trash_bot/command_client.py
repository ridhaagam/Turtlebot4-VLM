#!/usr/bin/env python3
"""
command_client.py
Lab3: ROS2 Python Service Client

This client sends commands to the command service.
Can be used by LLM or run standalone for testing.
"""

import sys
import json
import rclpy
from rclpy.node import Node
from trash_bot.srv import ExecuteCommand


class CommandClient(Node):
    """
    Client node to send commands to the robot.

    Usage:
        ros2 run trash_bot command_client.py search 30
        ros2 run trash_bot command_client.py forward 2
        ros2 run trash_bot command_client.py classify
        ros2 run trash_bot command_client.py stop
    """

    def __init__(self):
        super().__init__('command_client')

        self.client = self.create_client(ExecuteCommand, '/execute_command')

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for command service...')

        self.get_logger().info('Command client ready')

    def send_command(self, command: str, parameter: float = 0.0) -> dict:
        """
        Send a command to the robot.

        Args:
            command: Command string (search, approach, classify, forward, backward, left, right, stop)
            parameter: Optional parameter (duration, distance, etc.)

        Returns:
            dict with success, message, and result_data
        """
        request = ExecuteCommand.Request()
        request.command = command
        request.parameter = parameter

        self.get_logger().info(f'Sending command: {command} parameter={parameter}')

        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()

        response = {
            'success': result.success,
            'message': result.message,
            'data': json.loads(result.result_data) if result.result_data else {}
        }

        return response


def main(args=None):
    rclpy.init(args=args)

    client = CommandClient()

    # Parse command line arguments
    if len(sys.argv) < 2:
        print('Usage: ros2 run trash_bot command_client.py <command> [parameter]')
        print('')
        print('Commands:')
        print('  search [duration]   - Search for bins (default: 30s)')
        print('  approach [distance] - Approach detected bin (default: 0.5m)')
        print('  classify            - Classify bin fullness')
        print('  forward [duration]  - Move forward (default: 2s)')
        print('  backward [duration] - Move backward (default: 2s)')
        print('  left [duration]     - Turn left (default: 1s)')
        print('  right [duration]    - Turn right (default: 1s)')
        print('  stop                - Stop all movement')
        print('  return              - Return to start')
        print('')
        print('Examples:')
        print('  ros2 run trash_bot command_client.py search 60')
        print('  ros2 run trash_bot command_client.py forward 3')
        print('  ros2 run trash_bot command_client.py classify')

        client.destroy_node()
        rclpy.shutdown()
        return

    command = sys.argv[1]
    parameter = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0

    # Send command
    result = client.send_command(command, parameter)

    # Print result
    print('')
    print('=' * 50)
    print(f'Command: {command}')
    print(f'Parameter: {parameter}')
    print('=' * 50)
    print(f'Success: {result["success"]}')
    print(f'Message: {result["message"]}')
    if result['data']:
        print('Data:')
        for key, value in result['data'].items():
            print(f'  {key}: {value}')
    print('=' * 50)

    client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
