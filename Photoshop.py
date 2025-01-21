import os
import cv2
import numpy as np
from tkinter import Tk, filedialog
from PIL import Image, ImageOps

def select_folder():
    """ Opens a dialog to select a folder and returns the path. """
    root = Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(title="Select the folder with PNG files")
    return folder_selected if folder_selected else None

def add_noise(image, noise_level):
    """ Adds noise to an image using OpenCV. """
    noise = np.random.normal(0, noise_level, image.shape).astype(np.uint8)
    noisy_image = cv2.addWeighted(image, 1, noise, 0.2, 0)
    return noisy_image

def process_images(folder):
    """ Processes all PNG images in the selected folder. """
    output_main_folder = os.path.join(folder, "Processed_Images")
    os.makedirs(output_main_folder, exist_ok=True)

    for file_name in os.listdir(folder):
        if file_name.lower().endswith(".jpg"):
            file_path = os.path.join(folder, file_name)
            image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            
            if image is None:
                continue

            # Create an output folder for the image
            base_name = os.path.splitext(file_name)[0]
            output_folder = os.path.join(output_main_folder, base_name)
            os.makedirs(output_folder, exist_ok=True)

            # Save the original image
            original_save_path = os.path.join(output_folder, f"{base_name}.jpg")
            cv2.imwrite(original_save_path, image)

            # Generate noise variations
            for i in range(0, 110, 10):
                noisy_image = add_noise(image, i)
                noise_file_path = os.path.join(output_folder, f"{base_name}_noise_{i}.jpg")
                cv2.imwrite(noise_file_path, noisy_image)

                # Create rotated version
                flipped_image = cv2.rotate(noisy_image, cv2.ROTATE_180)
                flipped_file_path = os.path.join(output_folder, f"{base_name}_noise_Flipped_{i}.jpg")
                cv2.imwrite(flipped_file_path, flipped_image)

    print("Processing completed!")

if __name__ == "__main__":
    input_folder = select_folder()
    if input_folder:
        process_images(input_folder)
    else:
        print("No folder selected.")
