import os
import random
import shutil

def split_val_data(train_path, val_path, prefix, count):
    # Ensure the validation directory exists
    if not os.path.exists(val_path):
        os.makedirs(val_path)
        print(f"Created folder: {val_path}")

    # Get list of all images in the training folder
    all_images = [f for f in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, f))]
    
    if len(all_images) < count:
        print(f"Error: Not enough images in {train_path}. Found {len(all_images)}.")
        return

    # Randomly select 50 images
    selected_images = random.sample(all_images, count)
    print(f"Selected {count} random images to move...")

    for i, filename in enumerate(selected_images):
        # Full path for source
        src_file = os.path.join(train_path, filename)
        
        # New name and full path for destination (e.g., glioma-val001.jpg)
        new_name = f"{prefix}{str(i + 1).zfill(3)}.jpg"
        dest_file = os.path.join(val_path, new_name)
        
        # Move and rename
        shutil.move(src_file, dest_file)
        print(f"Moved & Renamed: {filename} -> {new_name}")

# Configuration
train_dir = r'D:\PROJECTS\Research\final\train\Normal'
val_dir = r'D:\PROJECTS\Research\final\val\Normal'
file_prefix = 'normal-val'
number_to_move = 50

split_val_data(train_dir, val_dir, file_prefix, number_to_move)
print("\nValidation split complete!")