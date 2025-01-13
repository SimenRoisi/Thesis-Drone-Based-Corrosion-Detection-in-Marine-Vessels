#!/usr/bin/env python3

import rospy
from image_processor import ImageProcessor
from voxel_mapper import VoxelMapper
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import MarkerArray
from cv_bridge import CvBridge
import numpy as np
import traceback
import time
from collections import deque
print(f"Loading {__file__}")
class CorrosionDetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_processor = ImageProcessor()
        self.voxel_mapper = VoxelMapper()
        
        self.latest_image = None
        self.latest_marker_msg = None
        self.camera_info = None

        self.image_sub = rospy.Subscriber('/l515/color/image_raw_with_header', Image, self.image_callback, queue_size=1)
        self.marker_sub = rospy.Subscriber('/occupied_cells_vis_array', MarkerArray, self.marker_callback, queue_size=1)
        self.camera_info_sub = rospy.Subscriber('/l515/color/camera_info', CameraInfo, self.camera_info_callback, queue_size=1)

        self.mask_visualization_pub = rospy.Publisher('/corrosion_mask_visualization', Image, queue_size=1)

        self.processing_times = deque(maxlen=100)  # Store last 100 processing times
        self.fps_update_interval = 10  # Update FPS every 10 frames
        self.frame_count = 0

        
        self.should_shutdown = False
        rospy.on_shutdown(self.shutdown_hook)
        
        rospy.loginfo("CorrosionDetector initialized successfully.")

    def shutdown_hook(self):
        self.should_shutdown = True
        rospy.loginfo("Shutting down CorrosionDetector...")

    def image_callback(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def marker_callback(self, msg):
        self.latest_marker_msg = msg

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def process_image(self):
            if self.latest_image is None or self.latest_marker_msg is None or self.camera_info is None:
                return

            start_time = time.time()
            image = self.latest_image
            marker_msg = self.latest_marker_msg
            camera_info = self.camera_info

            try:
                predicted_mask = self.image_processor.detect_corrosion(image)
                if predicted_mask is not None:
                    visualization = self.image_processor.create_mask_visualization(image, predicted_mask)
                    self.publish_mask_visualization(visualization)
                    self.voxel_mapper.update_voxel_map(predicted_mask, marker_msg, camera_info)
            except Exception as e:
                rospy.logerr(f"Error in image processing task: {e}")

            # Reset latest data
            self.latest_image = None
            self.latest_marker_msg = None
            self.camera_info = None

            end_time = time.time()
            processing_time = end_time - start_time
            self.processing_times.append(processing_time)

            self.frame_count += 1
            if self.frame_count % self.fps_update_interval == 0:
                self.log_performance_metrics()

    def log_performance_metrics(self):
        if len(self.processing_times) > 0:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            rospy.loginfo(f"Average processing time: {avg_time:.4f} seconds")
            rospy.loginfo(f"Estimated FPS: {fps:.2f}")

    def publish_mask_visualization(self, visualization):
        try:
            msg = self.bridge.cv2_to_imgmsg(visualization, "bgr8")
            self.mask_visualization_pub.publish(msg)
        except Exception as e:
            rospy.logerr(f"Error publishing mask visualization: {e}")

    def run(self):
            rate = rospy.Rate(10)  # 10 Hz
            while not rospy.is_shutdown() and not self.should_shutdown:
                self.process_image()
                rate.sleep()
            
            rospy.loginfo("CorrosionDetector has shut down.")