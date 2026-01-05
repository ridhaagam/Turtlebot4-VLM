# BinDetection.msg - Detection result for trash bins
std_msgs/Header header

bool bin_detected           # Whether a bin was detected
float32 distance            # Estimated distance to bin (meters)
float32 angle               # Angle to bin from robot center (radians)
string fullness_level       # "0-25%", "25-75%", "75-90%", "90-100%"
int32 fullness_percent      # Numeric fullness percentage
string label                # Detection label (e.g., "trash bin", "recycling bin")
float32 confidence          # Detection confidence (0.0 to 1.0)

# Primary bounding box in image coordinates (backward compatible)
int32 bbox_x
int32 bbox_y
int32 bbox_width
int32 bbox_height

# Multiple bounding boxes from Florence-2 detection
int32 num_detections        # Number of detected objects
string[] bbox_labels        # Labels for each detection
float32[] bbox_confidences  # Confidence for each detection
int32[] bbox_xs             # X coordinates
int32[] bbox_ys             # Y coordinates
int32[] bbox_widths         # Widths
int32[] bbox_heights        # Heights
