import os
import numpy as np

def process_label_files(directory):
    """
    Go through the directory, open each .label file, and convert all labels to uint32 format.
    """
    # Ensure the directory exists
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    # Get all .label files in the directory
    label_files = [f for f in os.listdir(directory) if f.endswith('.label')]

    if not label_files:
        print("No .label files found in the directory.")
        return

    print(f"Processing {len(label_files)} .label files...")

    for label_file in label_files:
        file_path = os.path.join(directory, label_file)
        print(f"Processing file: {label_file}")

        # Load the binary data from the file
        try:
            # Read the file as an array of uint32
            labels = np.fromfile(file_path, dtype=np.uint32)

            # Convert all labels to uint32 (this ensures dtype consistency)
            labels = labels.astype(np.uint32)

            # Write the converted data back to the file
            labels.tofile(file_path)
        except Exception as e:
            print(f"Failed to process file '{label_file}': {e}")

    print("Processing complete.")

# Specify the directory containing the .label files
directory_path = "/path/to/labels"

# Run the script
process_label_files(directory_path)
