#!/usr/bin/env python3
import numpy as np
import struct
import cv2
import math

# ------------------------- Distortion Function -------------------------
def distort_points(u, v, fx, fy, cx, cy, k1=-0.05, k2=0.0):
    """
    Applies a radial distortion to 2D points (u, v) using the simple model:
        x_d = x (1 + k1*r^2 + k2*r^4)
        y_d = y (1 + k1*r^2 + k2*r^4)
    where x, y are in normalized coordinates relative to (cx, cy).

    - k1 < 0 -> barrel distortion (edges outward)
    - k1 > 0 -> pincushion distortion (edges inward)

    Returns (u_dist, v_dist) as the distorted pixel coordinates.
    """
    # Convert (u,v) from pixel coords to normalized coords relative to principal point
    x_ = (u - cx) / fx
    y_ = (v - cy) / fy

    # Radius squared
    r2 = x_*x_ + y_*y_

    # Radial factor
    radial = 1.0 + k1 * r2 + k2 * (r2**2)

    # Distorted normalized
    x_d = x_ * radial
    y_d = y_ * radial

    # Map back to pixel
    u_d = x_d * fx + cx
    v_d = y_d * fy + cy
    return u_d, v_d


# ------------------------- Euler / Transform / IO Functions -------------------------
def euler_matrix_sxyz(roll, pitch, yaw):
    """
    Replicates tf_transformations.euler_matrix(roll, pitch, yaw, axes='sxyz')
    which corresponds to extrinsic rotations about X, then Y, then Z.
    Internally that is Rz(yaw) * Ry(pitch) * Rx(roll).

    Returns a 4x4 homogeneous rotation matrix with no translation.
    """
    Rx = np.array([
        [1,              0,               0],
        [0,  np.cos(roll),  -np.sin(roll)],
        [0,  np.sin(roll),   np.cos(roll)]
    ], dtype=np.float64)

    Ry = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [            0,  1,            0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ], dtype=np.float64)

    Rz = np.array([
        [ np.cos(yaw), -np.sin(yaw), 0],
        [ np.sin(yaw),  np.cos(yaw), 0],
        [          0,            0,  1]
    ], dtype=np.float64)

    # Multiply in order: Rz * Ry * Rx
    R = Rz @ Ry @ Rx

    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    return M

def load_velodyne_points(bin_path):
    """
    Reads a typical KITTI .bin file with floats (x, y, z, reflectance).
    Returns an Nx4 float32 array: [[x, y, z, r], ...].
    """
    points = []
    with open(bin_path, 'rb') as f:
        content = f.read()
        # Each point has 4 floats = 16 bytes
        it = struct.iter_unpack('ffff', content)
        for x, y, z, r in it:
            points.append((x, y, z, r))
    return np.array(points, dtype=np.float32)

def load_semantic_labels(label_path):
    """
    Reads a SemanticKITTI .label file. Each entry is a uint32 label.
    Returns an N array of labels.
    """
    return np.fromfile(label_path, dtype=np.uint32)

def parse_calib_file(calib_file):
    """
    Parse a typical KITTI-style calib.txt to extract:
      - P2 (3x4 camera projection for image_2)
      - Tr_velo_to_cam (4x4)
      - R0_rect or R_rect (4x4)
    """
    P2 = None
    Tr_velo_to_cam = None
    R_rect = np.eye(4, dtype=np.float64)  # default identity

    with open(calib_file, 'r') as f:
        for line in f:
            if line.startswith('P2:'):
                vals = line.strip().split()[1:]  # skip 'P2:'
                vals = list(map(float, vals))
                P2 = np.array(vals).reshape(3, 4)
            elif line.startswith('Tr_velo_to_cam:'):
                vals = line.strip().split()[1:]
                vals = list(map(float, vals))
                Tr_velo_to_cam = np.eye(4, dtype=np.float64)
                Tr_velo_to_cam[:3, :4] = np.array(vals).reshape(3, 4)
            elif line.startswith('R0_rect:') or line.startswith('R_rect'):
                vals = line.strip().split()[1:]
                vals = list(map(float, vals))
                R_mat = np.array(vals).reshape(3, 3)
                R_rect = np.eye(4, dtype=np.float64)
                R_rect[:3, :3] = R_mat

    if P2 is None or Tr_velo_to_cam is None:
        raise ValueError("Could not parse P2 or Tr_velo_to_cam from calib file.")
    
    return P2, Tr_velo_to_cam, R_rect

def get_color_for_label(label_id):
    """
    Very simple color mapping for demonstration: 
    Deterministically map label_id -> a BGR color.
    """
    np.random.seed(label_id)  # consistent color for same label
    color = np.random.randint(0, 255, (3,), dtype=np.uint8)
    return (int(color[0]), int(color[1]), int(color[2]))


# ------------------------- Main Script -------------------------
def main():
    # 1. File paths (adjust to match your setup)
    calib_path  = 'calib.txt'
    img_path    = 'image_2/00000.png'
    bin_path    = 'velodyne/00000.bin'
    label_path  = 'labels/00000.label'
    out_path    = 'overlay_00000.png'

    # 2. Load image
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    h, w = img.shape[:2]

    # 3. Parse calibration: P2 (3x4), Tr_velo_to_cam (4x4), R_rect (4x4)
    P2, Tr_velo_to_cam, R_rect = parse_calib_file(calib_path)

    # 4. Load point cloud + labels
    points = load_velodyne_points(bin_path)  # Nx4: x, y, z, reflectance
    labels = load_semantic_labels(label_path)  # N

    # 5. Convert to homogeneous Nx4
    N = points.shape[0]
    ones = np.ones((N, 1), dtype=np.float32)
    points_xyz1 = np.hstack((points[:, :3], ones))  # Nx4

    # 6. Optional: Build the alignment rotation to replicate tf_transformations
    #    euler_matrix(0, -pi/2, pi/2, axes='sxyz')
    R_align = euler_matrix_sxyz(0, -math.pi/2, math.pi/2)  # shape 4x4

    # 7. Combine with the LiDAR->Camera transform
    T_total = Tr_velo_to_cam @ R_align
    
    # 8. Transform points from LiDAR frame to "camera" frame
    points_cam = (T_total @ points_xyz1.T).T  # Nx4
    
    # 9. Optionally apply R_rect (if your data requires rectification)
    points_cam_rect = (R_rect @ points_cam.T).T  # Nx4

    # 10. Project via P2 (3x4)
    uvh = (P2 @ points_cam_rect.T).T  # Nx3
    u = uvh[:, 0] / uvh[:, 2]
    v = uvh[:, 1] / uvh[:, 2]
    z = uvh[:, 2]  # camera-plane depth

    # 11. Filter out invalid points
    valid = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u_valid = u[valid]
    v_valid = v[valid]
    lbl_valid = labels[valid]

    # 12. Distort the valid (u,v) to "bend" edges
    #     Extract fx, fy, cx, cy from the P2 matrix
    fx = P2[0,0]
    fy = P2[1,1]
    cx = P2[0,2]
    cy = P2[1,2]

    # Example radial distortion coefficients:
    # k1 negative => barrel distortion
    # Adjust to see how edges move. Start with small magnitude like -0.05 or -0.1.
    k1 = -0.05
    k2 = 0.0
    u_dist, v_dist = distort_points(u_valid, v_valid, fx, fy, cx, cy, k1, k2)

    # 13. Draw on image using the distorted coordinates
    for uu, vv, lbl in zip(u_dist, v_dist, lbl_valid):
        # Only draw if still in image bounds after distortion
        if 0 <= uu < w and 0 <= vv < h:
            color = get_color_for_label(lbl)
            cv2.circle(img, (int(uu), int(vv)), 2, color, -1)

    # 14. Save or display
    cv2.imwrite(out_path, img)
    print(f"Overlay saved to {out_path}")


if __name__ == '__main__':
    main()
