/**
 * motion_subscriber.cpp
 * ROS2 C++ Node for TurtleBot4 Lite movement control.
 *
 * This node subscribes to /robot_cmd topic and uses Create3 actions
 * (drive_distance, rotate_angle) to control the TurtleBot4 Lite.
 *
 * TurtleBot4 Lite uses actions, NOT /cmd_vel topic!
 */

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <cmath>
#include <mutex>
#include <thread>
#include <atomic>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "trash_bot/msg/robot_command.hpp"
#include "irobot_create_msgs/action/drive_distance.hpp"
#include "irobot_create_msgs/action/rotate_angle.hpp"

using namespace std::chrono_literals;
using std::placeholders::_1;
using std::placeholders::_2;

using DriveDistance = irobot_create_msgs::action::DriveDistance;
using RotateAngle = irobot_create_msgs::action::RotateAngle;
using GoalHandleDrive = rclcpp_action::ClientGoalHandle<DriveDistance>;
using GoalHandleRotate = rclcpp_action::ClientGoalHandle<RotateAngle>;

class MotionSubscriber : public rclcpp::Node
{
public:
    MotionSubscriber()
        : Node("motion_subscriber"),
          max_linear_speed_(0.15),   // m/s - safe speed for TurtleBot4 Lite
          max_angular_speed_(1.0),   // rad/s
          is_executing_(false)
    {
        // Declare parameters
        this->declare_parameter("max_linear_speed", 0.15);
        this->declare_parameter("max_angular_speed", 1.0);

        max_linear_speed_ = this->get_parameter("max_linear_speed").as_double();
        max_angular_speed_ = this->get_parameter("max_angular_speed").as_double();

        // Subscriber to receive robot commands
        command_subscription_ = this->create_subscription<trash_bot::msg::RobotCommand>(
            "/robot_cmd", 10,
            std::bind(&MotionSubscriber::command_callback, this, _1));

        // Action clients for TurtleBot4 Lite Create3 base
        drive_client_ = rclcpp_action::create_client<DriveDistance>(
            this, "/_do_not_use/drive_distance");
        rotate_client_ = rclcpp_action::create_client<RotateAngle>(
            this, "/_do_not_use/rotate_angle");

        RCLCPP_INFO(this->get_logger(), "Motion Subscriber Node started (Action-based)");
        RCLCPP_INFO(this->get_logger(), "Subscribing to: /robot_cmd");
        RCLCPP_INFO(this->get_logger(), "Using actions: drive_distance, rotate_angle");
        RCLCPP_INFO(this->get_logger(), "Max linear speed: %.2f m/s", max_linear_speed_);
        RCLCPP_INFO(this->get_logger(), "Max angular speed: %.2f rad/s", max_angular_speed_);

        // Wait for action servers
        wait_for_action_servers();
    }

private:
    void wait_for_action_servers()
    {
        RCLCPP_INFO(this->get_logger(), "Waiting for action servers...");

        while (!drive_client_->wait_for_action_server(std::chrono::seconds(2))) {
            if (!rclcpp::ok()) {
                RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for drive_distance action server");
                return;
            }
            RCLCPP_INFO(this->get_logger(), "Waiting for drive_distance action server...");
        }

        while (!rotate_client_->wait_for_action_server(std::chrono::seconds(2))) {
            if (!rclcpp::ok()) {
                RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for rotate_angle action server");
                return;
            }
            RCLCPP_INFO(this->get_logger(), "Waiting for rotate_angle action server...");
        }

        RCLCPP_INFO(this->get_logger(), "Action servers ready!");
    }

    void command_callback(const trash_bot::msg::RobotCommand::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "Received command: '%s' speed=%.2f duration=%.2f",
                    msg->command.c_str(), msg->speed, msg->duration);

        // Check if this is a command we should handle
        // Skip commands meant for other nodes (dock_now, resume_search, etc.)
        std::vector<std::string> motion_commands = {
            "forward", "backward", "left", "right", "stop", "search"
        };
        bool is_motion_command = false;
        for (const auto& cmd : motion_commands) {
            if (msg->command == cmd) {
                is_motion_command = true;
                break;
            }
        }

        if (!is_motion_command) {
            // Not a motion command - ignore it, let other nodes handle
            RCLCPP_DEBUG(this->get_logger(), "Ignoring non-motion command: '%s'", msg->command.c_str());
            return;
        }

        // If already executing, cancel and process new MOTION command
        if (is_executing_.load()) {
            RCLCPP_WARN(this->get_logger(), "Canceling previous command for new command");
            cancel_current_action();
        }

        // Clamp speed to [0, 1] range
        float speed_factor = std::max(0.0f, std::min(1.0f, msg->speed));
        float duration = msg->duration > 0.0f ? msg->duration : 2.0f;  // Default 2 seconds

        // Process command
        if (msg->command == "forward") {
            float distance = max_linear_speed_ * speed_factor * duration;
            RCLCPP_INFO(this->get_logger(), "Driving FORWARD %.2fm at %.2f m/s",
                        distance, max_linear_speed_ * speed_factor);
            send_drive_goal(distance, max_linear_speed_ * speed_factor);
        }
        else if (msg->command == "backward") {
            float distance = -max_linear_speed_ * speed_factor * duration;
            RCLCPP_INFO(this->get_logger(), "Driving BACKWARD %.2fm at %.2f m/s",
                        std::abs(distance), max_linear_speed_ * speed_factor);
            send_drive_goal(distance, max_linear_speed_ * speed_factor);
        }
        else if (msg->command == "left") {
            float angle = max_angular_speed_ * speed_factor * duration;
            RCLCPP_INFO(this->get_logger(), "Rotating LEFT %.2f rad at %.2f rad/s",
                        angle, max_angular_speed_ * speed_factor);
            send_rotate_goal(angle, max_angular_speed_ * speed_factor);
        }
        else if (msg->command == "right") {
            float angle = -max_angular_speed_ * speed_factor * duration;
            RCLCPP_INFO(this->get_logger(), "Rotating RIGHT %.2f rad at %.2f rad/s",
                        std::abs(angle), max_angular_speed_ * speed_factor);
            send_rotate_goal(angle, max_angular_speed_ * speed_factor);
        }
        else if (msg->command == "stop") {
            RCLCPP_INFO(this->get_logger(), "STOPPING - canceling current action");
            cancel_current_action();
        }
        else if (msg->command == "search") {
            // Search pattern: small forward movement
            float distance = max_linear_speed_ * 0.3 * speed_factor * duration;
            RCLCPP_INFO(this->get_logger(), "SEARCH mode: forward %.2fm", distance);
            send_drive_goal(distance, max_linear_speed_ * 0.3 * speed_factor);
        }
    }

    void send_drive_goal(float distance, float max_speed)
    {
        auto goal_msg = DriveDistance::Goal();
        goal_msg.distance = distance;
        goal_msg.max_translation_speed = max_speed;

        auto send_goal_options = rclcpp_action::Client<DriveDistance>::SendGoalOptions();
        send_goal_options.goal_response_callback =
            std::bind(&MotionSubscriber::drive_goal_response_callback, this, _1);
        send_goal_options.result_callback =
            std::bind(&MotionSubscriber::drive_result_callback, this, _1);

        is_executing_.store(true);
        drive_client_->async_send_goal(goal_msg, send_goal_options);
    }

    void send_rotate_goal(float angle, float max_speed)
    {
        auto goal_msg = RotateAngle::Goal();
        goal_msg.angle = angle;
        goal_msg.max_rotation_speed = max_speed;

        auto send_goal_options = rclcpp_action::Client<RotateAngle>::SendGoalOptions();
        send_goal_options.goal_response_callback =
            std::bind(&MotionSubscriber::rotate_goal_response_callback, this, _1);
        send_goal_options.result_callback =
            std::bind(&MotionSubscriber::rotate_result_callback, this, _1);

        is_executing_.store(true);
        rotate_client_->async_send_goal(goal_msg, send_goal_options);
    }

    void cancel_current_action()
    {
        if (current_drive_handle_) {
            drive_client_->async_cancel_goal(current_drive_handle_);
            current_drive_handle_ = nullptr;
        }
        if (current_rotate_handle_) {
            rotate_client_->async_cancel_goal(current_rotate_handle_);
            current_rotate_handle_ = nullptr;
        }
        is_executing_.store(false);
    }

    void drive_goal_response_callback(const GoalHandleDrive::SharedPtr & goal_handle)
    {
        if (!goal_handle) {
            RCLCPP_ERROR(this->get_logger(), "Drive goal was rejected");
            is_executing_.store(false);
        } else {
            RCLCPP_INFO(this->get_logger(), "Drive goal accepted");
            current_drive_handle_ = goal_handle;
        }
    }

    void drive_result_callback(const GoalHandleDrive::WrappedResult & result)
    {
        switch (result.code) {
            case rclcpp_action::ResultCode::SUCCEEDED:
                RCLCPP_INFO(this->get_logger(), "Drive completed successfully");
                break;
            case rclcpp_action::ResultCode::ABORTED:
                RCLCPP_WARN(this->get_logger(), "Drive was aborted");
                break;
            case rclcpp_action::ResultCode::CANCELED:
                RCLCPP_INFO(this->get_logger(), "Drive was canceled");
                break;
            default:
                RCLCPP_ERROR(this->get_logger(), "Drive unknown result code");
                break;
        }
        current_drive_handle_ = nullptr;
        is_executing_.store(false);
    }

    void rotate_goal_response_callback(const GoalHandleRotate::SharedPtr & goal_handle)
    {
        if (!goal_handle) {
            RCLCPP_ERROR(this->get_logger(), "Rotate goal was rejected");
            is_executing_.store(false);
        } else {
            RCLCPP_INFO(this->get_logger(), "Rotate goal accepted");
            current_rotate_handle_ = goal_handle;
        }
    }

    void rotate_result_callback(const GoalHandleRotate::WrappedResult & result)
    {
        switch (result.code) {
            case rclcpp_action::ResultCode::SUCCEEDED:
                RCLCPP_INFO(this->get_logger(), "Rotate completed successfully");
                break;
            case rclcpp_action::ResultCode::ABORTED:
                RCLCPP_WARN(this->get_logger(), "Rotate was aborted");
                break;
            case rclcpp_action::ResultCode::CANCELED:
                RCLCPP_INFO(this->get_logger(), "Rotate was canceled");
                break;
            default:
                RCLCPP_ERROR(this->get_logger(), "Rotate unknown result code");
                break;
        }
        current_rotate_handle_ = nullptr;
        is_executing_.store(false);
    }

    // Subscribers
    rclcpp::Subscription<trash_bot::msg::RobotCommand>::SharedPtr command_subscription_;

    // Action clients
    rclcpp_action::Client<DriveDistance>::SharedPtr drive_client_;
    rclcpp_action::Client<RotateAngle>::SharedPtr rotate_client_;

    // Current goal handles for cancellation
    GoalHandleDrive::SharedPtr current_drive_handle_;
    GoalHandleRotate::SharedPtr current_rotate_handle_;

    // Parameters
    double max_linear_speed_;
    double max_angular_speed_;

    // State
    std::atomic<bool> is_executing_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MotionSubscriber>());
    rclcpp::shutdown();
    return 0;
}
