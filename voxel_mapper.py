#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import ColorRGBA
import tf2_ros
import tf2_geometry_msgs

class CorrosionTracker:
    def __init__(self, corrosion_threshold=0.01, voxel_size=0.1, alpha=0.2):
        self.corrosion_map = {}
        self.last_update = {}
        self.corrosion_threshold = corrosion_threshold
        self.voxel_size = voxel_size
        self.alpha = alpha  # Smoothing factor for EMA
        self.total_area = 0
        self.corroded_area = 0

    def update_voxel(self, voxel_key, corrosion_class, current_time):
        if voxel_key not in self.corrosion_map:
            self.corrosion_map[voxel_key] = 0
            self.total_area += self.voxel_size ** 2

        previous_state = self.corrosion_map[voxel_key] in [1, 2, 3]
        current_state = corrosion_class in [1, 2, 3]

        self.corrosion_map[voxel_key] = corrosion_class

        if current_state and not previous_state:
            self.corroded_area += self.voxel_size ** 2
        elif not current_state and previous_state:
            self.corroded_area -= self.voxel_size ** 2

        self.last_update[voxel_key] = current_time

    def get_voxel_color(self, voxel_key):
        if voxel_key in self.corrosion_map:
            corrosion_class = self.corrosion_map[voxel_key]
            if corrosion_class == 1:  # Class 1 corrosion
                return ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)  # Red
            elif corrosion_class == 2:  # Class 2 corrosion
                return ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)  # Green
            elif corrosion_class == 3:  # Class 3 corrosion
                return ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)  # Blue
        
        # Default color for background (class 0) or any undefined class
        return ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)  # White

    def get_corrosion_percentage(self):
        if self.total_area == 0:
            return 0
        return (self.corroded_area / self.total_area) * 100
    
    def get_total_area(self):
        return self.total_area
    
    def get_corroded_area(self):
        return self.corroded_area

    def get_corrosion_status(self):
        percentage = self.get_corrosion_percentage()
        if percentage < 2:
            return "Good"
        elif percentage < 20:
            return "Fair"
        else:
            return "Bad"

class VoxelMapper:
    def __init__(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.corrosion_voxel_pub = rospy.Publisher('/corrosion_voxel_map', MarkerArray, queue_size=1)
        self.voxel_size = rospy.get_param('~voxel_size', 0.1)
        self.corrosion_tracker = CorrosionTracker(voxel_size=self.voxel_size, alpha=0.1)  # Initialize with alpha for EMA

    def update_voxel_map(self, predicted_mask, marker_msg, camera_info):
        rospy.loginfo("=== BEGIN update_voxel_map ===")
        height, width = predicted_mask.shape
        fx, fy, cx, cy = camera_info.K[0], camera_info.K[4], camera_info.K[2], camera_info.K[5]

        new_marker_array = MarkerArray()
        current_time = rospy.Time.now()

        for marker in marker_msg.markers:
            new_marker = self._create_new_marker(marker)

            for point in marker.points:
                voxel_key = self._get_voxel_key(point)
                transformed_point = self._transform_point(point, marker.header.frame_id)
                if transformed_point and self._is_point_in_image(transformed_point, fx, fy, cx, cy, height, width):
                    corrosion_class = self._get_corrosion_value(transformed_point, predicted_mask, fx, fy, cx, cy)
                    self.corrosion_tracker.update_voxel(voxel_key, corrosion_class, current_time)

                color = self.corrosion_tracker.get_voxel_color(voxel_key)
                new_marker.points.append(point)
                new_marker.colors.append(color)

            new_marker_array.markers.append(new_marker)

        self.corrosion_voxel_pub.publish(new_marker_array)
        rospy.loginfo("=== END update_voxel_map ===")
        self._log_corrosion_stats()

    def _create_new_marker(self, original_marker):
        new_marker = Marker()
        new_marker.header = original_marker.header
        new_marker.ns = "corrosion_voxel_map"
        new_marker.id = original_marker.id
        new_marker.type = original_marker.type
        new_marker.action = original_marker.action
        new_marker.scale = original_marker.scale
        new_marker.pose = original_marker.pose
        return new_marker

    def _get_voxel_key(self, point):
        return tuple(np.round(np.array([point.x, point.y, point.z]) / self.voxel_size).astype(int))

    def _transform_point(self, point, frame_id):
        try:
            point_stamped = PointStamped()
            point_stamped.header.frame_id = frame_id
            point_stamped.point = point
            transform = self.tf_buffer.lookup_transform('l515_color_optical_frame', 
                                                        frame_id, 
                                                        rospy.Time(0), 
                                                        rospy.Duration(0.1))
            transformed_point = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
            return transformed_point.point
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Could not transform point: {e}")
            return None

    def _is_point_in_image(self, point, fx, fy, cx, cy, height, width):
        x, y, z = point.x, point.y, point.z
        if z <= 0:
            return False
        u = int((x * fx / z) + cx)
        v = int((y * fy / z) + cy)
        return 0 <= u < width and 0 <= v < height

    def _get_corrosion_value(self, point, predicted_mask, fx, fy, cx, cy):
        x, y, z = point.x, point.y, point.z
        u = int((x * fx / z) + cx)
        v = int((y * fy / z) + cy)
        
        # Check neighboring pixels
        for du in [-1, 0, 1]:
            for dv in [-1, 0, 1]:
                nu, nv = u + du, v + dv
                if 0 <= nu < predicted_mask.shape[1] and 0 <= nv < predicted_mask.shape[0]:
                    value = predicted_mask[nv, nu]
                    if value > 0:
                        return value
        
        return 0  # Return 0 if no corrosion found in the neighborhood

    def _log_corrosion_stats(self):
        total_voxels = len(self.corrosion_tracker.corrosion_map)
        corroded_voxels = sum(1 for v in self.corrosion_tracker.corrosion_map.values() if v > self.corrosion_tracker.corrosion_threshold)
        corrosion_percentage = (corroded_voxels / total_voxels * 100) if total_voxels > 0 else 0
        
        rospy.loginfo(f"Corrosion Stats: Total Voxels: {total_voxels}, "
                      f"Corroded Voxels: {corroded_voxels}, "
                      f"Corrosion Percentage: {corrosion_percentage:.2f}%")