"""DINOv2 binary road segmentation for rover_road_assist.

camera Image -> segmentation_node -> /rover/perception/road_mask (mono8, 0/1)
             -> rover_road_assist/road_deviation_node -> /rover/road/state
"""
