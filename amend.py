import os
import numpy as np

def process_label_files(directory):
    """
    Go through the directory, open each .label file, and replace label 9 with 51.
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
            labels = np.fromfile(file_path, dtype=np.uint32)

            # Replace occurrences of 9 with 51
            labels[labels == 9] = 51

            # Write the modified data back to the file
            labels.tofile(file_path)
        except Exception as e:
            print(f"Failed to process file '{label_file}': {e}")

    print("Processing complete.")

# Specify the directory containing the .label files
directory_path = "/path/to/labels"

# Run the script
process_label_files(directory_path)
