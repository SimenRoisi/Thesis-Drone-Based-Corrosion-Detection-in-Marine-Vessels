# From Pixels to Persistent Voxels: Drone-Based Corrosion Detection in Marine Vessels

This repository contains the code and resources for the master's thesis: **"From Pixels to Persistent Voxels: Drone-Based Corrosion Detection in Marine Vessels using Deep Learning and 3D Voxel Mapping"**. The thesis explores an innovative approach to automated corrosion detection and 3D mapping using drones equipped with RGB cameras and LiDAR sensors.

## Overview

Corrosion poses significant risks to the maritime industry, leading to safety hazards and financial losses. This project introduces a novel, drone-based system that combines deep learning for 2D corrosion detection with 3D spatial mapping for persistent and accurate corrosion analysis in tanker ships.

Key features:
- **Corrosion Detection**: Uses UNet-based deep learning models with various encoder backbones (MobileNetV3 and ResNet) for pixel-wise segmentation of corrosion in RGB images.
- **3D Mapping**: Integrates corrosion detections into a 3D point cloud generated from LiDAR data, utilizing OctoMap for efficient voxel-based representation.
- **Real-Time Processing**: Achieves efficient, real-time processing suitable for deployment on drones.

## System Highlights

- **Dataset**: Included 556 annotated images of corrosion with preprocessing for standardization and augmentation to improve model performance.
- **Models**: UNet architectures with MobileNetV3 and ResNet encoders were tested for accuracy and computational efficiency.
- **Results**: MobileNetV3-Large achieved the best balance between detection accuracy (79% Corrosion Detection Rate) and computational efficiency, making it ideal for drone-based applications.
- **Visualization**: Outputs corrosion detections as persistent 3D voxel maps, viewable in ROS RViz.
