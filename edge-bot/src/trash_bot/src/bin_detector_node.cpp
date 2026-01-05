/**
 * bin_detector_node.cpp
 * Lab3: ROS2 C++ Publisher
 *
 * This node subscribes to camera image topic and publishes bin detection results.
 * Uses simple image processing to detect potential bin shapes.
 * Full classification is done by the Python classifier_node using Florence-2.
 */

#include <memory>
#include <chrono>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/bool.hpp"
#include "trash_bot/msg/bin_detection.hpp"
#include "cv_bridge/cv_bridge.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

using namespace std::chrono_literals;
using std::placeholders::_1;

class BinDetectorNode : public rclcpp::Node
{
public:
    BinDetectorNode()
        : Node("bin_detector_node"),
          frame_count_(0),
          detection_count_(0)
    {
        // Declare parameters
        this->declare_parameter("camera_topic", "/camera/image_raw");
        this->declare_parameter("min_contour_area", 5000);
        this->declare_parameter("detection_interval", 5);  // Process every Nth frame

        camera_topic_ = this->get_parameter("camera_topic").as_string();
        min_contour_area_ = this->get_parameter("min_contour_area").as_int();
        detection_interval_ = this->get_parameter("detection_interval").as_int();

        // Publisher for bin detection results
        detection_publisher_ = this->create_publisher<trash_bot::msg::BinDetection>(
            "/bin_detected", 10);

        // Publisher for simple detection flag (for other nodes to trigger actions)
        detection_flag_publisher_ = this->create_publisher<std_msgs::msg::Bool>(
            "/bin_detection_flag", 10);

        // Subscriber to camera image
        image_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            camera_topic_, 10,
            std::bind(&BinDetectorNode::image_callback, this, _1));

        RCLCPP_INFO(this->get_logger(), "Bin Detector Node started");
        RCLCPP_INFO(this->get_logger(), "Subscribing to: %s", camera_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "Publishing to: /bin_detected, /bin_detection_flag");
    }

private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        frame_count_++;

        // Process only every Nth frame to reduce CPU load
        if (frame_count_ % detection_interval_ != 0) {
            return;
        }

        try {
            // Convert ROS image to OpenCV
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
            cv::Mat frame = cv_ptr->image;

            // Detect potential bins using simple shape detection
            auto detections = detect_bin_shapes(frame);

            // Publish results
            for (const auto& det : detections) {
                // Publish full detection message
                detection_publisher_->publish(det);

                // Publish detection flag
                auto flag_msg = std_msgs::msg::Bool();
                flag_msg.data = true;
                detection_flag_publisher_->publish(flag_msg);

                detection_count_++;
                RCLCPP_INFO(this->get_logger(),
                            "Potential bin detected at (%d, %d) size %dx%d",
                            det.bbox_x, det.bbox_y, det.bbox_width, det.bbox_height);
            }

            if (detections.empty() && frame_count_ % 50 == 0) {
                RCLCPP_DEBUG(this->get_logger(), "No bins detected in frame %ld", frame_count_);
            }

        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }

    std::vector<trash_bot::msg::BinDetection> detect_bin_shapes(const cv::Mat& frame)
    {
        std::vector<trash_bot::msg::BinDetection> detections;

        // Convert to grayscale
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Apply Gaussian blur
        cv::Mat blurred;
        cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

        // Edge detection
        cv::Mat edges;
        cv::Canny(blurred, edges, 50, 150);

        // Dilate to close gaps
        cv::Mat dilated;
        cv::dilate(edges, dilated, cv::Mat(), cv::Point(-1, -1), 2);

        // Find contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(dilated, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        int image_center_x = frame.cols / 2;

        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);

            // Filter by minimum area
            if (area < min_contour_area_) {
                continue;
            }

            // Get bounding rectangle
            cv::Rect bbox = cv::boundingRect(contour);

            // Filter by aspect ratio (bins are typically taller than wide, or roughly square)
            float aspect_ratio = static_cast<float>(bbox.width) / bbox.height;
            if (aspect_ratio < 0.3 || aspect_ratio > 2.0) {
                continue;
            }

            // Filter by size relative to image
            float size_ratio = static_cast<float>(bbox.area()) / (frame.cols * frame.rows);
            if (size_ratio < 0.01 || size_ratio > 0.5) {
                continue;
            }

            // Create detection message
            auto det = trash_bot::msg::BinDetection();
            det.header.stamp = this->now();
            det.header.frame_id = "camera_frame";

            det.bin_detected = true;
            det.bbox_x = bbox.x;
            det.bbox_y = bbox.y;
            det.bbox_width = bbox.width;
            det.bbox_height = bbox.height;

            // Estimate distance based on bbox height (rough approximation)
            // Assumes a bin of ~0.5m height at 2m distance takes ~200px
            det.distance = (200.0 * 2.0) / bbox.height;

            // Estimate angle from center of image
            int bbox_center_x = bbox.x + bbox.width / 2;
            float pixel_offset = bbox_center_x - image_center_x;
            float fov_horizontal = 60.0 * M_PI / 180.0;  // Assuming 60 degree FOV
            det.angle = (pixel_offset / image_center_x) * (fov_horizontal / 2);

            // Initial values (will be updated by classifier)
            det.label = "potential_bin";
            det.confidence = 0.5;  // Low confidence for shape-based detection
            det.fullness_level = "unknown";
            det.fullness_percent = -1;

            detections.push_back(det);
        }

        return detections;
    }

    // Subscribers and Publishers
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
    rclcpp::Publisher<trash_bot::msg::BinDetection>::SharedPtr detection_publisher_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr detection_flag_publisher_;

    // Parameters
    std::string camera_topic_;
    int min_contour_area_;
    int detection_interval_;

    // State
    long frame_count_;
    long detection_count_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<BinDetectorNode>());
    rclcpp::shutdown();
    return 0;
}
