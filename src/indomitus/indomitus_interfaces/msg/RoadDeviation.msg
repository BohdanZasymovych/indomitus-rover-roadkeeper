std_msgs/Header header        # stamp copied from the mask
float32 lateral_offset        # [-1..1], + = road centre is to the right of the vehicle, in road half-widths
float32 heading_error         # rad, + = road bends to the right
float32 confidence            # [0..1]
bool    valid                 # false = no usable road in view
