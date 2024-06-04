import os
import json
from PIL import Image

def resize_and_pad(image, target_size, base_aspect_ratio):
    original_ratio = image.size[0] / image.size[1]
    if original_ratio == base_aspect_ratio:
        resized_image = image.resize(target_size, Image.ANTIALIAS)
    else:
        ratio = min(target_size[0] / image.size[0], target_size[1] / image.size[1])
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        resized_image = image.resize(new_size, Image.ANTIALIAS)
        new_img = Image.new("RGB", target_size, (0, 0, 0))
        new_img.paste(resized_image, ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2))
        resized_image = new_img
    return resized_image, new_size

def process_images(input_directory, output_directory, target_size=(1280, 960), base_aspect_ratio=4/3):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for filename in os.listdir(input_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            image_path = os.path.join(input_directory, filename)
            with Image.open(image_path) as img:
                img_processed, new_size = resize_and_pad(img, target_size, base_aspect_ratio)
                img_processed.save(os.path.join(output_directory, filename))
    return new_size

def update_annotations(original_json_path, new_json_path, input_directory, output_directory, target_size=(1280, 960), base_aspect_ratio=4/3):
    with open(original_json_path, 'r') as f:
        data = json.load(f)
    
    new_size = process_images(input_directory, output_directory, target_size, base_aspect_ratio)
    
    scale_x = new_size[0] / target_size[0]
    scale_y = new_size[1] / target_size[1]
    pad_x = (target_size[0] - new_size[0]) // 2
    pad_y = (target_size[1] - new_size[1]) // 2
    
    for annotation in data['annotations']:
        bbox = annotation['bbox']
        annotation['bbox'] = [
            bbox[0] * scale_x + pad_x,
            bbox[1] * scale_y + pad_y,
            bbox[2] * scale_x,
            bbox[3] * scale_y
        ]
        
        if 'segmentation' in annotation:
            for seg in annotation['segmentation']:
                for i in range(0, len(seg), 2):
                    seg[i] = seg[i] * scale_x + pad_x
                    seg[i + 1] = seg[i + 1] * scale_y + pad_y
    
    with open(new_json_path, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    input_directory = 'inspectrone/inspectrone_train'  # Change to your input folder path
    output_directory = 'train_images'  # Change to your output folder path
    original_json_path = 'inspectrone_train.json'  # Path to the original JSON file
    new_json_path = 'inspectrone_train_updated.json'  # Path to save the new JSON file
    
    update_annotations(original_json_path, new_json_path, input_directory, output_directory)

if __name__ == "__main__":
    main()
