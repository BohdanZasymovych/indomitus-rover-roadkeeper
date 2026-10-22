# Waiter-FPV-drone threat from rover_drone_detection, one per processed frame.
uint8 CLEAR=0                 # nothing confirmed
uint8 WARN=1                  # a confirmed drone in view
uint8 DANGER=2                # a confirmed drone close, or on the path ahead

std_msgs/Header header        # stamp and frame_id of the camera image
uint8   level
uint16  count                 # detections in this frame
float32 confidence            # [0..1] score of the most threatening detection; 0 when none
float32 bearing               # rad, REP-103: + = left of the camera axis
float32 size                  # its box height / image height (bigger = closer)
float32 range_est             # m, rough, from the apparent size; NaN when unknown
