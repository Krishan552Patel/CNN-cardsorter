import os

def list_empty_folders(directory):
    empty_folders = []
    
    for root, dirs, files in os.walk(directory):
        # Check if the folder has no files
        if not files and not any(os.path.isdir(os.path.join(root, d)) for d in dirs):
            empty_folders.append(root)
    
    return empty_folders

# Example usage
directory_path = "E:/card/Processed_Imagese/FAI010"
empty_folders = list_empty_folders(directory_path)

if empty_folders:
    print("Empty folders found:")
    for folder in empty_folders:
        print(folder)
else:
    print("No empty folders found.")
