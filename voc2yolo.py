# from pylabel import importer
#
# dataset = importer.ImportVOC(path=r"C:\Users\Kenny\PycharmProjects\yolo_bear\train2\label_voc")
# dataset.export.ExportToYoloV5()

import os

# Folder containing the label files
label_folder = r"label folder"

# Loop through the specific file range
for i in range(152, 200):  # from screen_152.txt to screen_199.txt
    file_path = os.path.join(label_folder, f"screen_{i}.txt")

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}, skipping...")
        continue

    # Open the file, read content, replace '0' with '1' for the class ID
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Replace class ID '0' with '1'
    new_lines = [line.replace('0 ', '1 ', 1) for line in lines]

    # Save the changes back to the file
    with open(file_path, 'w') as file:
        file.writelines(new_lines)

print("Class ID update complete!")

