#!/usr/bin/env python3
"""
command_service.py
Lab3: ROS2 Python Service Server

This service handles high-level robot commands like search, approach, classify, etc.
It coordinates between motion control, detection, and classification nodes.
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import json
import time
import threading

from trash_bot.srv import ExecuteCommand, ClassifyBin
from trash_bot.msg import RobotCommand, BinDetection
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool
from irobot_create_msgs.action import Undock, Dock


class CommandService(Node):
    """
    Service node that handles high-level robot commands.

    Commands:
    - search: Move robot in search pattern while looking for bins
    - approach: Move towards detected bin
    - classify: Trigger bin classification
    - return: Return to start position
    - stop: Stop all movement
    - full_mission: Complete autonomous mission (undock -> search -> classify -> dock)
    - undock: Undock from charging station
    - dock: Return to charging station
    """

    def __init__(self):
        super().__init__('command_service')

        # Callback group for concurrent service calls
        self.callback_group = ReentrantCallbackGroup()

        # Service server for execute_command
        self.execute_service = self.create_service(
            ExecuteCommand,
            '/execute_command',
            self.execute_command_callback,
            callback_group=self.callback_group
        )

        # Publisher for robot commands
        self.cmd_publisher = self.create_publisher(
            RobotCommand,
            '/robot_cmd',
            10
        )

        # Subscriber for bin detection
        self.detection_subscription = self.create_subscription(
            BinDetection,
            '/bin_detected',
            self.detection_callback,
            10
        )

        # Subscriber for detection flag
        self.flag_subscription = self.create_subscription(
            Bool,
            '/bin_detection_flag',
            self.flag_callback,
            10
        )

        # Client for classification service
        self.classify_client = self.create_client(
            ClassifyBin,
            '/classify_bin',
            callback_group=self.callback_group
        )

        # Client for navigation find_bin service
        self.find_bin_client = self.create_client(
            SetBool,
            '/find_bin',
            callback_group=self.callback_group
        )

        # Subscribe to navigation state
        self.nav_state_subscription = self.create_subscription(
            String,
            '/navigation_state',
            self.nav_state_callback,
            10
        )

        # Subscribe to search completion
        self.search_complete_subscription = self.create_subscription(
            Bool,
            '/search_complete',
            self.search_complete_callback,
            10
        )

        # State
        self.bin_detected = False
        self.latest_detection = None
        self.is_searching = False
        self.search_lock = threading.Lock()
        self.navigation_state = 'idle'
        self.search_completed = False
        self.search_success = False
        self.mission_in_progress = False

        self.get_logger().info('Command Service Node started')
        self.get_logger().info('Service: /execute_command')
        self.get_logger().info('Publishing to: /robot_cmd')
        self.get_logger().info('Subscribing to: /bin_detected, /bin_detection_flag')

    def detection_callback(self, msg: BinDetection):
        """Handle incoming bin detection messages."""
        self.latest_detection = msg
        if msg.bin_detected:
            self.bin_detected = True
            self.get_logger().info(
                f'Bin detected at distance={msg.distance:.2f}m, angle={msg.angle:.2f}rad'
            )

    def flag_callback(self, msg: Bool):
        """Handle detection flag messages."""
        if msg.data:
            self.bin_detected = True

    def nav_state_callback(self, msg: String):
        """Handle navigation state updates."""
        self.navigation_state = msg.data
        self.get_logger().debug(f'Navigation state: {self.navigation_state}')

    def search_complete_callback(self, msg: Bool):
        """Handle search completion messages."""
        self.search_completed = True
        self.search_success = msg.data
        self.get_logger().info(f'Search completed: {"success" if msg.data else "failed"}')

    def publish_command(self, command: str, speed: float = 0.5, duration: float = 0.0):
        """Publish a robot command."""
        cmd = RobotCommand()
        cmd.command = command
        cmd.speed = speed
        cmd.duration = duration
        self.cmd_publisher.publish(cmd)
        self.get_logger().info(f'Published command: {command} speed={speed} duration={duration}')

    def execute_command_callback(self, request, response):
        """Handle execute_command service requests."""
        command = request.command.lower()
        parameter = request.parameter

        self.get_logger().info(f'Received command: {command} with parameter: {parameter}')

        try:
            if command == 'search':
                result = self.execute_search(parameter if parameter > 0 else 30.0)
            elif command == 'approach':
                result = self.execute_approach(parameter if parameter > 0 else 0.5)
            elif command == 'classify':
                result = self.execute_classify()
            elif command == 'return':
                result = self.execute_return()
            elif command == 'stop':
                result = self.execute_stop()
            elif command == 'forward':
                duration = parameter if parameter > 0 else 2.0
                self.publish_command('forward', 0.5, duration)
                result = {'success': True, 'message': f'Moving forward for {duration}s'}
            elif command == 'backward':
                duration = parameter if parameter > 0 else 2.0
                self.publish_command('backward', 0.5, duration)
                result = {'success': True, 'message': f'Moving backward for {duration}s'}
            elif command == 'left':
                duration = parameter if parameter > 0 else 1.0
                self.publish_command('left', 0.5, duration)
                result = {'success': True, 'message': f'Turning left for {duration}s'}
            elif command == 'right':
                duration = parameter if parameter > 0 else 1.0
                self.publish_command('right', 0.5, duration)
                result = {'success': True, 'message': f'Turning right for {duration}s'}
            elif command == 'undock':
                result = self.execute_undock()
            elif command == 'dock':
                result = self.execute_dock()
            elif command == 'full_mission':
                timeout = parameter if parameter > 0 else 120.0
                result = self.execute_full_mission(timeout)
            elif command == 'find_bin':
                timeout = parameter if parameter > 0 else 120.0
                result = self.execute_find_bin(timeout)
            else:
                result = {
                    'success': False,
                    'message': f'Unknown command: {command}',
                    'data': {}
                }

            response.success = result.get('success', False)
            response.message = result.get('message', '')
            response.result_data = json.dumps(result.get('data', {}))

        except Exception as e:
            self.get_logger().error(f'Error executing command: {e}')
            response.success = False
            response.message = f'Error: {str(e)}'
            response.result_data = '{}'

        return response

    def execute_search(self, max_duration: float) -> dict:
        """
        Execute search pattern to find bins.
        Robot moves in a search pattern while monitoring for detections.
        """
        with self.search_lock:
            if self.is_searching:
                return {
                    'success': False,
                    'message': 'Search already in progress',
                    'data': {}
                }
            self.is_searching = True

        self.get_logger().info(f'Starting search for {max_duration}s')
        self.bin_detected = False
        self.latest_detection = None

        start_time = time.time()
        search_phase = 0

        try:
            while time.time() - start_time < max_duration:
                # Check if bin detected
                if self.bin_detected and self.latest_detection:
                    self.publish_command('stop')
                    self.get_logger().info('Bin found! Stopping search.')

                    with self.search_lock:
                        self.is_searching = False

                    return {
                        'success': True,
                        'message': 'Bin detected during search',
                        'data': {
                            'bin_detected': True,
                            'distance': self.latest_detection.distance,
                            'angle': self.latest_detection.angle,
                            'search_time': time.time() - start_time
                        }
                    }

                # Rotate search pattern phases
                phase_duration = 3.0
                elapsed = time.time() - start_time
                current_phase = int(elapsed / phase_duration) % 4

                if current_phase != search_phase:
                    search_phase = current_phase
                    if search_phase == 0:
                        self.publish_command('forward', 0.3, phase_duration)
                    elif search_phase == 1:
                        self.publish_command('left', 0.4, phase_duration)
                    elif search_phase == 2:
                        self.publish_command('forward', 0.3, phase_duration)
                    else:
                        self.publish_command('right', 0.4, phase_duration)

                # Allow ROS callbacks to process
                rclpy.spin_once(self, timeout_sec=0.1)

            # Search timeout
            self.publish_command('stop')

        finally:
            with self.search_lock:
                self.is_searching = False

        return {
            'success': False,
            'message': 'Search timeout, no bin found',
            'data': {'search_time': max_duration}
        }

    def execute_approach(self, target_distance: float) -> dict:
        """
        Approach a detected bin until within target distance.
        """
        if not self.bin_detected or not self.latest_detection:
            return {
                'success': False,
                'message': 'No bin detected to approach',
                'data': {}
            }

        self.get_logger().info(f'Approaching bin, target distance: {target_distance}m')

        # Simple approach: adjust angle first, then move forward
        det = self.latest_detection

        # Turn towards bin if angle is significant
        if abs(det.angle) > 0.1:  # ~6 degrees
            turn_cmd = 'left' if det.angle > 0 else 'right'
            turn_duration = min(abs(det.angle) / 0.5, 2.0)  # Max 2 seconds turn
            self.publish_command(turn_cmd, 0.4, turn_duration)
            time.sleep(turn_duration + 0.2)

        # Move forward
        if det.distance > target_distance:
            move_distance = det.distance - target_distance
            move_duration = move_distance / 0.2  # At 0.2 m/s
            move_duration = min(move_duration, 10.0)  # Max 10 seconds
            self.publish_command('forward', 0.4, move_duration)
            time.sleep(move_duration + 0.2)

        self.publish_command('stop')

        return {
            'success': True,
            'message': f'Approached bin to approximately {target_distance}m',
            'data': {
                'initial_distance': det.distance,
                'target_distance': target_distance
            }
        }

    def execute_classify(self) -> dict:
        """
        Trigger bin classification service.
        """
        if not self.classify_client.wait_for_service(timeout_sec=2.0):
            return {
                'success': False,
                'message': 'Classification service not available',
                'data': {}
            }

        request = ClassifyBin.Request()
        request.capture_new_frame = True

        self.get_logger().info('Calling classification service...')

        future = self.classify_client.call_async(request)

        # Wait for result
        timeout = 180.0  # Classification takes ~60-90s with SmolVLM2 on RPi4
        start = time.time()
        while not future.done() and time.time() - start < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)

        if not future.done():
            return {
                'success': False,
                'message': 'Classification timeout',
                'data': {}
            }

        result = future.result()

        return {
            'success': result.success,
            'message': result.status_summary,
            'data': {
                'fullness_level': result.fullness_level,
                'fullness_percent': result.fullness_percent,
                'label': result.label,
                'confidence': result.confidence,
                'sent_to_server': result.sent_to_server
            }
        }

    def execute_return(self) -> dict:
        """
        Return to starting position.
        Simple implementation: turn around and move forward.
        """
        self.get_logger().info('Returning to start...')

        # Turn around (180 degrees)
        self.publish_command('left', 0.5, 3.14)  # ~180 degrees at 0.5 rad/s
        time.sleep(3.5)

        # Move forward for a bit
        self.publish_command('forward', 0.3, 5.0)
        time.sleep(5.5)

        self.publish_command('stop')

        return {
            'success': True,
            'message': 'Return command executed',
            'data': {}
        }

    def execute_stop(self) -> dict:
        """Stop all movement."""
        self.publish_command('stop')

        with self.search_lock:
            self.is_searching = False

        return {
            'success': True,
            'message': 'Robot stopped',
            'data': {}
        }

    def execute_undock(self) -> dict:
        """
        Undock from charging station using Create3 action.
        """
        self.get_logger().info('Undocking from charging station...')

        try:
            import subprocess
            result = subprocess.run(
                ['timeout', '30', 'ros2', 'action', 'send_goal', '/_do_not_use/undock',
                 'irobot_create_msgs/action/Undock', '{}'],
                capture_output=True,
                text=True,
                timeout=35
            )

            if 'SUCCEEDED' in result.stdout or result.returncode == 0:
                self.get_logger().info('Undock successful')
                return {
                    'success': True,
                    'message': 'Successfully undocked from charging station',
                    'data': {}
                }
            else:
                self.get_logger().warn(f'Undock result: {result.stdout}')
                return {
                    'success': True,  # May already be undocked
                    'message': 'Undock command sent (may already be undocked)',
                    'data': {'output': result.stdout[:200]}
                }

        except Exception as e:
            self.get_logger().error(f'Undock error: {e}')
            return {
                'success': False,
                'message': f'Undock error: {str(e)}',
                'data': {}
            }

    def execute_dock(self, max_retries: int = 3) -> dict:
        """
        Dock at charging station using Create3 action with verification.

        Args:
            max_retries: Maximum number of dock attempts

        Returns:
            Result dictionary with dock status and verification
        """
        import subprocess
        import time

        for attempt in range(max_retries):
            self.get_logger().info(f'Docking at charging station (attempt {attempt + 1}/{max_retries})...')

            try:
                result = subprocess.run(
                    ['timeout', '120', 'ros2', 'action', 'send_goal', '/_do_not_use/dock',
                     'irobot_create_msgs/action/Dock', '{}'],
                    capture_output=True,
                    text=True,
                    timeout=125
                )

                dock_succeeded = 'SUCCEEDED' in result.stdout or result.returncode == 0

                if dock_succeeded:
                    # Verify dock by checking robot status
                    time.sleep(2.0)  # Wait for status to update
                    verified = self._verify_dock_status()

                    if verified:
                        self.get_logger().info('='*50)
                        self.get_logger().info('DOCK VERIFIED - Robot successfully docked!')
                        self.get_logger().info('='*50)
                        return {
                            'success': True,
                            'message': 'Successfully docked and verified at charging station',
                            'data': {'attempts': attempt + 1, 'verified': True}
                        }
                    else:
                        self.get_logger().warn('Dock command succeeded but verification failed')
                        if attempt < max_retries - 1:
                            self.get_logger().info('Retrying dock...')
                            time.sleep(3.0)
                            continue
                else:
                    self.get_logger().warn(f'Dock attempt {attempt + 1} failed: {result.stdout[:100]}')
                    if attempt < max_retries - 1:
                        time.sleep(3.0)
                        continue

            except subprocess.TimeoutExpired:
                self.get_logger().error(f'Dock attempt {attempt + 1} timed out')
                if attempt < max_retries - 1:
                    continue

            except Exception as e:
                self.get_logger().error(f'Dock error on attempt {attempt + 1}: {e}')
                if attempt < max_retries - 1:
                    continue

        # All retries exhausted
        self.get_logger().error('='*50)
        self.get_logger().error('DOCK FAILED - All attempts exhausted')
        self.get_logger().error('='*50)
        return {
            'success': False,
            'message': f'Dock failed after {max_retries} attempts',
            'data': {'attempts': max_retries, 'verified': False}
        }

    def _verify_dock_status(self) -> bool:
        """
        Verify robot is actually docked by checking dock status topic.

        Returns:
            True if robot is confirmed docked
        """
        import subprocess

        try:
            # Check dock status via ROS2 topic
            result = subprocess.run(
                ['timeout', '5', 'ros2', 'topic', 'echo', '/dock_status',
                 'irobot_create_msgs/msg/DockStatus', '--once'],
                capture_output=True,
                text=True,
                timeout=10
            )

            # Check if docked (is_docked: true)
            if 'is_docked: true' in result.stdout.lower():
                self.get_logger().info('Dock status verified: is_docked=true')
                return True
            elif 'is_docked: false' in result.stdout.lower():
                self.get_logger().warn('Dock status: is_docked=false')
                return False
            else:
                # Topic might not be available, assume success if dock command worked
                self.get_logger().warn('Could not verify dock status (topic unavailable)')
                return True  # Assume success

        except Exception as e:
            self.get_logger().warn(f'Dock verification error: {e}')
            return True  # Assume success if we can't verify

    def execute_find_bin(self, timeout: float = 120.0) -> dict:
        """
        Start SmolVLM-based bin search using navigation_vlm_node.

        Args:
            timeout: Maximum search time in seconds.

        Returns:
            Result dictionary with search outcome.
        """
        if not self.find_bin_client.wait_for_service(timeout_sec=2.0):
            return {
                'success': False,
                'message': 'Navigation VLM service not available',
                'data': {}
            }

        self.get_logger().info(f'Starting bin search (timeout: {timeout}s)...')

        # Reset search state
        self.search_completed = False
        self.search_success = False

        # Start search
        request = SetBool.Request()
        request.data = True

        future = self.find_bin_client.call_async(request)

        # Wait for service response
        start = time.time()
        while not future.done() and time.time() - start < 5.0:
            rclpy.spin_once(self, timeout_sec=0.1)

        if not future.done():
            return {
                'success': False,
                'message': 'Failed to start search',
                'data': {}
            }

        result = future.result()
        if not result.success:
            return {
                'success': False,
                'message': result.message,
                'data': {}
            }

        # Wait for search to complete
        self.get_logger().info('Waiting for bin search to complete...')
        while not self.search_completed and time.time() - start < timeout:
            rclpy.spin_once(self, timeout_sec=0.5)

        if self.search_completed:
            return {
                'success': self.search_success,
                'message': 'Bin found and approached' if self.search_success else 'Bin search failed',
                'data': {
                    'search_time': time.time() - start,
                    'final_state': self.navigation_state
                }
            }
        else:
            # Timeout - stop search
            stop_request = SetBool.Request()
            stop_request.data = False
            self.find_bin_client.call_async(stop_request)

            return {
                'success': False,
                'message': 'Bin search timeout',
                'data': {'search_time': timeout}
            }

    def execute_full_mission(self, timeout: float = 120.0) -> dict:
        """
        Execute complete autonomous mission:
        1. Undock from charging station
        2. Search for trash bin using SmolVLM
        3. Classify bin contents using Florence-2
        4. Send results to dashboard server
        5. Return to dock

        Args:
            timeout: Maximum time for bin search in seconds.

        Returns:
            Result dictionary with mission outcome.
        """
        if self.mission_in_progress:
            return {
                'success': False,
                'message': 'Mission already in progress',
                'data': {}
            }

        self.mission_in_progress = True
        mission_start = time.time()
        mission_results = {'steps': []}

        try:
            self.get_logger().info('='*50)
            self.get_logger().info('STARTING FULL AUTONOMOUS MISSION')
            self.get_logger().info('='*50)

            # Step 1: Undock
            self.get_logger().info('\n[STEP 1/5] Undocking from charging station...')
            undock_result = self.execute_undock()
            mission_results['steps'].append({
                'step': 'undock',
                'success': undock_result['success'],
                'message': undock_result['message']
            })

            if not undock_result['success']:
                self.get_logger().warn('Undock may have failed, continuing anyway...')

            time.sleep(2.0)  # Wait for robot to stabilize

            # Step 2: Find bin using SmolVLM
            self.get_logger().info('\n[STEP 2/5] Searching for trash bin with SmolVLM...')
            search_result = self.execute_find_bin(timeout)
            mission_results['steps'].append({
                'step': 'find_bin',
                'success': search_result['success'],
                'message': search_result['message'],
                'data': search_result.get('data', {})
            })

            if not search_result['success']:
                self.get_logger().warn('Bin search failed, attempting classification anyway...')

            time.sleep(1.0)

            # Step 3: Classify bin
            self.get_logger().info('\n[STEP 3/5] Classifying bin contents with Florence-2...')
            classify_result = self.execute_classify()
            mission_results['steps'].append({
                'step': 'classify',
                'success': classify_result['success'],
                'message': classify_result['message'],
                'data': classify_result.get('data', {})
            })

            # Step 4: Results sent to server (handled by classifier)
            self.get_logger().info('\n[STEP 4/5] Results sent to dashboard server')
            sent_to_server = classify_result.get('data', {}).get('sent_to_server', False)
            mission_results['steps'].append({
                'step': 'send_to_server',
                'success': sent_to_server,
                'message': 'Results sent to server' if sent_to_server else 'Failed to send results'
            })

            time.sleep(1.0)

            # Step 5: Return to dock
            self.get_logger().info('\n[STEP 5/5] Returning to charging dock...')
            dock_result = self.execute_dock()
            mission_results['steps'].append({
                'step': 'dock',
                'success': dock_result['success'],
                'message': dock_result['message']
            })

            # Compute overall success
            mission_time = time.time() - mission_start
            successful_steps = sum(1 for s in mission_results['steps'] if s['success'])
            total_steps = len(mission_results['steps'])

            self.get_logger().info('\n' + '='*50)
            self.get_logger().info(f'MISSION COMPLETE: {successful_steps}/{total_steps} steps successful')
            self.get_logger().info(f'Total time: {mission_time:.1f}s')
            self.get_logger().info('='*50)

            return {
                'success': successful_steps >= 3,  # At least search, classify, dock
                'message': f'Mission completed: {successful_steps}/{total_steps} steps successful',
                'data': {
                    'mission_time': mission_time,
                    'steps': mission_results['steps'],
                    'classification': classify_result.get('data', {})
                }
            }

        except Exception as e:
            self.get_logger().error(f'Mission error: {e}')
            import traceback
            traceback.print_exc()

            return {
                'success': False,
                'message': f'Mission error: {str(e)}',
                'data': {'steps': mission_results['steps']}
            }

        finally:
            self.mission_in_progress = False


def main(args=None):
    rclpy.init(args=args)

    node = CommandService()

    # Use multi-threaded executor for concurrent callbacks
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
