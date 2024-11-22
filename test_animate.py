import numpy as np
import open3d as o3d
import os
import cv2

# Custom label-to-color mapping
CUSTOM_LABEL_COLORS = {
    1: [245, 150, 100],   # Containers
    2: [255, 0, 0],       # Road
    3: [30, 30, 255],     # Person
    4: [255, 255, 0],     # Crane
    5: [0, 255, 0],       # Hangar
    6: [200, 200, 200],   # Short Tower
    7: [150, 0, 150],     # Large Tower
    8: [0, 255, 255],     # Slender Tower
    9: [255, 128, 0],     # Construction Tower
    10: [0, 128, 0],      # Terrain
    11: [128, 128, 128],  # Storage Bin
    12: [0, 0, 128],      # Veer Right Sign
    13: [128, 0, 0],      # Do Not Enter Sign
    14: [128, 128, 0],    # Speed Limit 15 Sign
    15: [0, 128, 128],    # Road Closed Sign
    16: [255, 0, 255],    # Stop Sign
    17: [150, 75, 0],     # Car
    18: [255, 192, 203],  # Person-Adult
    19: [255, 223, 186],  # Person-Child
}

def quaternion_to_matrix(quaternion):
    """Convert a quaternion into a 4x4 homogeneous transformation matrix."""
    x, y, z, w = quaternion
    n = x * x + y * y + z * z + w * w
    if n < np.finfo(float).eps:
        return np.eye(4)  # Identity matrix if quaternion is zero

    s = 2.0 / n
    xs, ys, zs = x * s, y * s, z * s
    wx, wy, wz = w * xs, w * ys, w * zs
    xx, xy, xz = x * xs, x * ys, x * zs
    yy, yz, zz = y * ys, y * zs, z * zs

    rotation_matrix = np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy, 0.0],
        [xy + wz, 1.0 - (xx + zz), yz - wx, 0.0],
        [xz - wy, yz + wx, 1.0 - (xx + yy), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    return rotation_matrix

def get_r_align():
    """Construct the alignment matrix R_align manually."""
    R_y = np.array([
        [np.cos(-np.pi / 2), 0, np.sin(-np.pi / 2)],
        [0, 1, 0],
        [-np.sin(-np.pi / 2), 0, np.cos(-np.pi / 2)]
    ])

    R_z = np.array([
        [np.cos(np.pi / 2), -np.sin(np.pi / 2), 0],
        [np.sin(np.pi / 2), np.cos(np.pi / 2), 0],
        [0, 0, 1]
    ])

    R_flip = np.diag([1, -1, -1])  # Flip to correct z-axis alignment

    R_align = R_flip @ R_z @ R_y  # Apply flip after rotations

    R_align_homogeneous = np.eye(4)
    R_align_homogeneous[:3, :3] = R_align
    return R_align_homogeneous

def transform_points(lidar_points, quaternion):
    """Transform lidar points to the camera frame."""
    T_total = quaternion_to_matrix(quaternion) @ get_r_align()  # Combine transformations
    lidar_points_homogeneous = np.hstack((lidar_points, np.ones((lidar_points.shape[0], 1))))  # Nx4
    points_camera_frame = (T_total @ lidar_points_homogeneous.T).T  # Transform points
    return points_camera_frame[:, :3]  # Return Nx3 points

def project_points(pc, calib):
    """Project 3D points onto 2D image space using calibration parameters."""
    P = calib['P0']  # 3x4 camera intrinsic matrix
    Tr_velo_to_cam = calib['Tr_velo_to_cam']  # 3x4 transformation matrix

    # Transform to the camera coordinate frame
    pc_homogeneous = np.hstack((pc, np.ones((pc.shape[0], 1))))  # Nx4
    pc_cam = (Tr_velo_to_cam @ pc_homogeneous.T).T

    # Filter points behind the camera
    pc_cam = pc_cam[pc_cam[:, 2] > 0]

    # Project to image space
    points_2d = (P @ pc_cam.T).T
    points_2d[:, :2] /= points_2d[:, 2:]  # Normalize by depth

    return points_2d[:, :2], pc_cam[:, 2]  # Return 2D points and depth

def overlay_points_on_image(image, points_2d, labels, depth):
    """Overlay 3D points onto a 2D image."""
    overlay = image.copy()
    for i, point in enumerate(points_2d):
        x, y = int(point[0]), int(point[1])
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:  # Check bounds
            color = CUSTOM_LABEL_COLORS.get(labels[i], [255, 255, 255])  # Default white for unknown
            depth_normalized = int(255 * (1 - min(1, depth[i] / 50.0)))  # Normalize depth for brightness
            cv2.circle(overlay, (x, y), 2, [c * depth_normalized / 255 for c in color], -1)
    return overlay

def main(dataset_path, sequence="00"):
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
        bin_path = os.path.join(velodyne_path, bin_file)
        pc = load_point_cloud(bin_path)
        
        labels = None
        if label_files:
            label_file = os.path.join(label_path, bin_file.replace(".bin", ".label"))
            if os.path.exists(label_file):
                labels = load_labels(label_file)

        # Load image
        img_file = os.path.join(image_path, image_files[i])
        image = cv2.imread(img_file)

        # Project points onto image
        pc_transformed = transform_points(pc, [0, 0, 0, 1])  # Replace with actual quaternion
        points_2d, depth = project_points(pc_transformed, calib)

        # Overlay points
        overlay = overlay_points_on_image(image, points_2d, labels, depth)
        cv2.imshow("Overlay", overlay)
        if cv2.waitKey(0) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cv2.destroyAllWindows()

def load_calibration(calib_path):
    """Load calibration data from a KITTI format calibration file."""
    calib = {}
    with open(calib_path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            calib[key] = np.array([float(x) for x in value.split()]).reshape(-1, 4)
    return calib
