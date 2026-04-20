import os
from PIL import Image

def rename_and_convert_images(folder_path, prefix, start_num, end_num):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    print(f"Found {len(files)} files. Starting conversion...")

    for i, filename in enumerate(files):
        if i >= (end_num - start_num + 1):
            break
            
        try:

            file_path = os.path.join(folder_path, filename)
            
            img = Image.open(file_path)
            
            rgb_img = img.convert('RGB')
            
            new_name = f"{prefix}{str(i + start_num).zfill(3)}.jpg"
            new_path = os.path.join(folder_path, new_name)
            
  
            rgb_img.save(new_path, 'JPEG')
            
            if file_path != new_path:
                os.remove(file_path)
                
            print(f"Converted: {filename} -> {new_name}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

target_folder = r'D:\PROJECTS\Research\final\train\Pituitary' #change target folder for each class (pituitary is just an example) 
file_prefix = 'pituitary-train' # change -train , -val , -test according to need 
start_count = 1
end_count = 5000 # every class has images less than 5000

rename_and_convert_images(target_folder, file_prefix, start_count, end_count)
print("Finished!")
