import os
import numpy as np
from PIL import Image, ImageDraw
import torch
import json
from torch.utils.data import Dataset

class Dataclass(Dataset):
    def __init__(self, image_dir, annotation_file):
        self.image_dir = image_dir
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        self.images = self.annotations['images']
        self.anns = {ann['image_id']: [] for ann in self.annotations['annotations']}
        for ann in self.annotations['annotations']:
            self.anns[ann['image_id']].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_path = os.path.join(self.image_dir, image_info['file_name'])


        image = Image.open(image_path).convert('RGB')
        image = np.array(image, dtype=np.float32)

        anns = self.anns[image_info['id']]
        mask_original = Image.new('L', (image_info['width'], image_info['height']), 0)
        for ann in anns:
            if 'segmentation' in ann:
                polygon = ann['segmentation'][0]
                class_id = ann['category_id']
                ImageDraw.Draw(mask_original).polygon(polygon, outline=class_id, fill=class_id)

        # Create a new mask with padding
        mask_padded = Image.new('L', (image_info['width'], image_info['height'] + 8), 0)
        # Paste the original mask onto the padded mask, aligning it in the center
        mask_padded.paste(mask_original, (0, 4))

        mask = np.array(mask_padded)

        # Convert to PyTorch tensors
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        image = image.permute(2, 0, 1)  # Change [height, width, channels] to [channels, height, width]

        return image, mask