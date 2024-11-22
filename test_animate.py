import numpy as np
import open3d as o3d
import cv2
import os

def load_point_cloud(bin_path):
    """Load a point cloud from a binary .bin file."""
    pc = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)  # x, y, z, intensity
    return pc[:, :3]  # Only use x, y, z

def load_labels(label_path):
    """Load labels from a .label file."""
    labels = np.fromfile(label_path, dtype=np.uint32)
    return labels & 0xFFFF  # Semantic labels (first 16 bits)

def load_calibration(calib_path):
    """Load calibration data from a KITTI format calibration file."""
    calib = {}
    with open(calib_path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            calib[key] = np.array([float(x) for x in value.split()]).reshape(-1, 4)
    return calib
def get_r_align():
    """Construct the alignment matrix R_align manually."""
    # Rotation around the y-axis by -π/2
    R_y = np.array([
        [np.cos(-np.pi / 2), 0, np.sin(-np.pi / 2)],
        [0, 1, 0],
        [-np.sin(-np.pi / 2), 0, np.cos(-np.pi / 2)]
    ])

    # Rotation around the z-axis by π/2
    R_z = np.array([
        [np.cos(np.pi / 2), -np.sin(np.pi / 2), 0],
        [np.sin(np.pi / 2), np.cos(np.pi / 2), 0],
        [0, 0, 1]
    ])

    # Combined rotation: R_align = R_z @ R_y
    R_align = R_z @ R_y
    return R_align
def project_points(pc, calib):
    """Project 3D points onto 2D image space using calibration parameters."""
    # Load calibration matrices
    P = calib['P0']  # 3x4 camera intrinsic matrix
    Tr_velo_to_cam = calib['Tr_velo_to_cam']  # 3x4 transformation matrix

    # Get alignment matrix
    R_align = get_r_align()

    # Align point cloud to match camera coordinate system
    pc_aligned = (R_align @ pc.T).T

    # Convert point cloud to homogeneous coordinates
    pc_homogeneous = np.hstack((pc_aligned, np.ones((pc_aligned.shape[0], 1))))  # Nx4

    # Transform points to the camera coordinate frame
    pc_cam = (Tr_velo_to_cam @ pc_homogeneous.T).T  # Nx4 -> Nx3

    # Filter points behind the camera
    pc_cam = pc_cam[pc_cam[:, 2] > 0]

    # Project points to image space
    pc_image = (P @ pc_cam.T).T  # Nx4 -> Nx3
    pc_image[:, :2] /= pc_image[:, 2:]  # Normalize by depth (z)

    return pc_image[:, :2], pc_cam[:, 2]  # Return 2D points and depth

def overlay_points_on_image(image, points_2d, labels, depth, category_color_map):
    """Overlay 3D points onto a 2D image."""
    overlay = image.copy()
    for i, point in enumerate(points_2d):
        x, y = int(point[0]), int(point[1])
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:  # Check bounds
            color = category_color_map.get(labels[i], [255, 255, 255])  # Default white for unknown
            depth_normalized = int(255 * (1 - min(1, depth[i] / 50.0)))  # Normalize depth for brightness
            cv2.circle(overlay, (x, y), 2, [c * depth_normalized / 255 for c in color], -1)
    return overlay

def visualize_overlay(dataset_path, sequence="00", frame_rate=10):
    """Animate point cloud projections over image frames."""
    sequence_path = os.path.join(dataset_path, "sequences", sequence)
    velodyne_path = os.path.join(sequence_path, "velodyne")
    label_path = os.path.join(sequence_path, "labels")
    image_path = os.path.join(sequence_path, "image_2")
    calib_path = os.path.join(sequence_path, "calib.txt")

    # Load calibration data
    calib = load_calibration(calib_path)

    # List files in the sequence
    bin_files = sorted(os.listdir(velodyne_path))
    label_files = sorted(os.listdir(label_path)) if os.path.exists(label_path) else None
    image_files = sorted(os.listdir(image_path))

    for i, bin_file in enumerate(bin_files):
        # Load point cloud and labels
        bin_path = os.path.join(velodyne_path, bin_file)
        pc = load_point_cloud(bin_path)
        
        labels = None
        if label_files:
            label_file = os.path.join(label_path, bin_file.replace(".bin", ".label"))
            if os.path.exists(label_file):
                labels = load_labels(label_file)
        
        # Project points onto image
        points_2d, depth = project_points(pc, calib)
        
        # Load corresponding image
        img_file = os.path.join(image_path, bin_file.replace(".bin", ".png"))
        image = cv2.imread(img_file)
        
        # Overlay points
        overlay = overlay_points_on_image(image, points_2d, labels, depth, CATEGORY_COLOR_MAP)

        # Display overlay
        cv2.imshow("Overlay", overlay)
        if cv2.waitKey(int(1000 / frame_rate)) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    dataset_path = "path/to/your/dataset"
    visualize_overlay(dataset_path, sequence="00", frame_rate=10)
