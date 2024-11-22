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
def project_and_overlay_points(lidar_points, calib, image):
    """
    Transform lidar points to the camera frame, project to 2D, and overlay on the image.
    """
    # Step 1: Align the lidar points
    R_align = get_r_align()
    lidar_points_aligned = (R_align[:3, :3] @ lidar_points.T).T

    # Step 2: Transform to the camera frame
    pc_homogeneous = np.hstack((lidar_points_aligned, np.ones((lidar_points_aligned.shape[0], 1))))  # Nx4
    pc_cam = (calib['Tr_velo_to_cam'] @ pc_homogeneous.T).T

    # Step 3: Filter points behind the camera
    pc_cam = pc_cam[pc_cam[:, 2] > 0]

    # Step 4: Project points to image space
    points_2d = (calib['P0'] @ pc_cam.T).T
    points_2d[:, :2] /= points_2d[:, 2:]  # Normalize by depth

    # Step 5: Overlay points on the image
    overlay = image.copy()
    for point in points_2d:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:  # Check bounds
            cv2.circle(overlay, (x, y), 2, (0, 255, 0), -1)  # Draw green points
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
