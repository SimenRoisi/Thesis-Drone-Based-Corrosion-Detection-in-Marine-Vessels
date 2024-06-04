import os
from PIL import Image

def resize_and_pad(image, target_size, base_aspect_ratio):
    # Calculate the new size maintaining the aspect ratio
    original_ratio = image.size[0] / image.size[1]
    if original_ratio == base_aspect_ratio:
        # Resize directly to the target size if it matches the base aspect ratio
        resized_image = image.resize(target_size, Image.ANTIALIAS)
    else:
        # Calculate new size that maintains aspect ratio
        ratio = min(target_size[0] / image.size[0], target_size[1] / image.size[1])
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        resized_image = image.resize(new_size, Image.ANTIALIAS)
        # Create a new image with white background and paste the resized image
        new_img = Image.new("RGB", target_size, (0, 0, 0))
        new_img.paste(resized_image, ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2))
        resized_image = new_img

    return resized_image

def process_images(input_directory, output_directory, target_size=(1280, 960), base_aspect_ratio=4/3):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for filename in os.listdir(input_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            image_path = os.path.join(input_directory, filename)
            with Image.open(image_path) as img:
                # Determine aspect ratio and apply appropriate resizing and padding
                img_processed = resize_and_pad(img, target_size, base_aspect_ratio)
                # Save the processed image, consider changing the path as needed
                img_processed.save(os.path.join(output_directory, filename))

def main():
    # Set the directories here
    input_directory = 'inspectrone/inspectrone_train'  # Change to your input folder path
    output_directory = 'train_images'  # Change to your output folder path
    
    process_images(input_directory, output_directory)

if __name__ == "__main__":
    main()
