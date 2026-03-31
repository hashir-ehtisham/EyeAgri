import os
import random
import shutil

def split_data(main_path, destination_path, prefix, count):
    # Ensure the validation directory exists
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
        print(f"Created folder: {destination_path}")

    # Get list of all images in the training folder
    all_images = [f for f in os.listdir(main_path) if os.path.isfile(os.path.join(main_path, f))]
    
    if len(all_images) < count:
        print(f"Error: Not enough images in {main_path}. Found {len(all_images)}.")
        return

    # Randomly select images
    selected_images = random.sample(all_images, count)
    print(f"Selected {count} random images to move...")

    for i, filename in enumerate(selected_images):
        # Full path for source
        src_file = os.path.join(main_path, filename)
        
        # New name and full path for destination (e.g., glioma-val001.jpg)
        new_name = f"{prefix}{str(i + 1).zfill(3)}.jpg"
        dest_file = os.path.join(destination_path, new_name)
        
        # Move and rename
        shutil.move(src_file, dest_file)
        print(f"Moved & Renamed: {filename} -> {new_name}")

# Configuration
train_dir = r'D:\PROJECTS\tst\final\train\Glioma' # Path to main directory (train from where we are splitting)
destination_dir = r'D:\PROJECTS\tst\final\val\Glioma' # Path to destination directory
file_prefix = 'classname-val' # change classname to class and val to test according to need
number_to_move = 120 # do 60 for test

split_data(train_dir, destination_dir, file_prefix, number_to_move)
print("\nValidation split complete!")
