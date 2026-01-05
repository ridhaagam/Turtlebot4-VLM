# RobotCommand.msg - Command for robot movement
string command              # forward, backward, left, right, stop, search
float32 speed               # Speed (0.0 to 1.0, maps to actual velocity)
float32 duration            # Duration in seconds (0 = continuous until stop)
