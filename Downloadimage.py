import csv
import os
import requests

# Define file paths
csv_file_path = "card.csv"  # Change this if your CSV file is in a different location
output_folder = "E:/Pic"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Read the CSV file
with open(csv_file_path, "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row

    for row in reader:
        unique_id, card_id, image_url, foiling, edition = row

        # Ensure proper naming for the file
        foiling = foiling.replace(" ", "_")  # Replace spaces with underscores
        edition = edition.replace(" ", "_")  # Replace spaces with underscores

        # Construct the file name
        filename = f"{card_id}-{foiling}-{edition}.jpg"
        filepath = os.path.join(output_folder, filename)

        # Download the image
        try:
            response = requests.get(image_url, stream=True)
            response.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)

            # Save the image
            with open(filepath, "wb") as img_file:
                for chunk in response.iter_content(1024):
                    img_file.write(chunk)
            
            print(f"Downloaded: {filename}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {image_url}: {e}")

print("Download complete.")
