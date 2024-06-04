import os
from PIL import Image

def get_image_size_distribution(directory):
    size_count = {}
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            image_path = os.path.join(directory, filename)
            with Image.open(image_path) as img:
                size = img.size  # size is a tuple (width, height)
                if size in size_count:
                    size_count[size] += 1
                else:
                    size_count[size] = 1
    return size_count

def main():
    # Set the directory path to your folder containing images
    directory = 'inspectrone/inspectrone_train'
    size_distribution = get_image_size_distribution(directory)
    print("Image size distribution in the folder:")
    for size, count in size_distribution.items():
        print(f"Size: {size}, Count: {count}")

if __name__ == "__main__":
    main()
