#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
import os
print(f"Loading {__file__}")
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from unet_mobileL import UNet

class ImageProcessor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.transform = self._create_transform()

    def _load_model(self):
        torch.cuda.empty_cache()
        model_path = rospy.get_param('~model_path', '/home/autostudent/catkin_ws/src/corrosion_detection/scripts/unet_MobLarge-7.pth')
        model = UNet(n_classes=4, dropout_prob=0.0).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def _create_transform(self):
        return A.Compose([
            A.Resize(height=512, width=512),
            #A.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0, value=(0, 0, 0)),
            ToTensorV2()
        ])

    def preprocess_image(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_normalized = image_rgb.astype(np.float32) / 255.0
        transformed = self.transform(image=image_normalized)
        return transformed['image'].unsqueeze(0).to(self.device)

    def detect_corrosion(self, image):
        try:
            input_tensor = self.preprocess_image(image)
            with torch.no_grad():
                output = self.model(input_tensor)
            predicted_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            return predicted_mask
        except Exception as e:
            rospy.logerr(f"Error in corrosion detection: {e}")
            return None

    def create_mask_visualization(self, original_image, predicted_mask):
        orig_h, orig_w = original_image.shape[:2]
        resized_mask = cv2.resize(predicted_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        color_mask = np.zeros_like(original_image)
        color_mask[resized_mask == 1] = [0, 0, 255]    # Red for class 1
        color_mask[resized_mask == 2] = [0, 255, 0]    # Green for class 2
        color_mask[resized_mask == 3] = [255, 0, 0]    # Blue for class 3
        return cv2.addWeighted(original_image, 1, color_mask, 0.5, 0)