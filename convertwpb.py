import os
from PIL import Image

# Define the folder containing the .webp files and the output folder for the .png files
input_folder = 'E:CARDSpics/CARDdb/ENGLISH/HH'
output_folder = 'E:CARDSpics/CARDdb/ENGLISH/HH'

# Create the output folder if it doesn't exist


# Loop through each file in the input folder
#for filename in os.listdir(input_folder):
   
#print("done")
        
def strip_filename_suffix(folder_path, suffix):
    """
    Strips everything in the filename starting from the given suffix.

    Parameters:
    folder_path (str): Path to the folder containing the files.
    suffix (str): The suffix to strip from filenames.
    """
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            base_name = os.path.splitext(filename)[0]
            if suffix in base_name:
                new_base_name = base_name.split(suffix)[0]
                new_filename = new_base_name + '.png'
                os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
                print(f'Renamed {filename} to {new_filename}')
                
strip_filename_suffix(input_folder, '.width')