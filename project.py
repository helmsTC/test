import numpy as np
import cv2
import struct

def parse_calib_file(calib_path, cam_idx=2):
    """
    Parses a KITTI-style calib.txt, returning the 3x4 P matrix for the
    specified camera index (e.g., P2 for `image_2`).
    """
    # Example lines in calib.txt:
    # P0: 7.215377e+02 ...
    # P1: ...
    # P2: ...
    # R0_rect: ...
    # Tr_velo_to_cam: ...
    #
    # We'll just extract the 'P2:' line by default.

    camera_string = f'P{cam_idx}:'
    P = None

    with open(calib_path, 'r') as f:
        for line in f:
            if camera_string in line:
                # Strip 'P2:' then split by space
                raw_data = line.strip().split(' ')
                # raw_data[0] is 'P2:', rest are the matrix numbers
                mat_values = raw_data[1:]  # everything after 'P2:'
                mat_values = [float(v) for v in mat_values]
                P = np.reshape(mat_values, (3, 4))
                break

    if P is None:
        raise ValueError(f"Could not find {camera_string} in {calib_path}")
    return P

def load_velodyne_points(bin_path):
    """
    Reads a KITTI-style .bin file with floats (x, y, z, reflectance) per point.
    Returns an Nx4 numpy array.
    """
    points = []
    with open(bin_path, 'rb') as f:
        content = f.read()
        # Each point is 4 floats = 16 bytes
        # # of points = len(content) / 16
        # Use struct.unpack to parse all at once
        pts_iter = struct.iter_unpack('ffff', content)
        for x, y, z, r in pts_iter:
            points.append((x, y, z, r))
    points = np.array(points, dtype=np.float32)
    return points

def load_semantic_labels(label_path):
    """
    Reads a SemanticKITTI .label file.
    Each entry is a uint32 label. 
    Returns an N array of labels.
    """
    labels = np.fromfile(label_path, dtype=np.uint32)
    return labels

def get_color_for_label(label_id):
    """
    A simple color map for demonstration: 
    returns a BGR color tuple for a given label_id.
    You can customize as needed.
    """
    # Quick hack: map label to a consistent color
    # e.g. use the label ID bits or a small LUT
    np.random.seed(label_id)
    color = np.random.randint(0, 255, (3,), dtype=np.uint8)
    return (int(color[0]), int(color[1]), int(color[2]))

def main():
    # 1) File paths
    calib_file = 'calib.txt'
    image_file = 'image_2/00000.png'
    velodyne_file = 'velodyne/00000.bin'
    label_file = 'labels/00000.label'

    # 2) Parse calibration
    P = parse_calib_file(calib_file, cam_idx=2)  # for image_2
    # P is shape (3,4)

    # 3) Load image
    img = cv2.imread(image_file)
    if img is None:
        raise FileNotFoundError(f"Could not load {image_file}")
    height, width = img.shape[:2]

    # 4) Load point cloud + labels
    points = load_velodyne_points(velodyne_file)  # Nx4
    labels = load_semantic_labels(label_file)     # N

    # 5) Convert points to homogeneous, Nx4
    #    We only need x,y,z for projection. reflectance is points[:,3]
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    points_xyz1 = np.hstack((points[:, :3], ones))  # Nx4

    # 6) Because Unreal is left-handed and typical KITTI / OpenCV is right-handed,
    #    we often see a mirrored projection. One quick fix is to flip the x-axis:
    flip_x = np.diag([-1, 1, 1, 1]).astype(np.float32)  # shape (4,4)
    points_xyz1_flipped = (flip_x @ points_xyz1.T).T

    # 7) Project with the camera matrix P (3x4)
    #    2D homogeneous coords = P * (x,y,z,1)
    #    Then convert to pixel coords: (u = x'/z', v = y'/z')
    points_2d = (P @ points_xyz1_flipped.T).T  # Nx3
    u = points_2d[:, 0] / points_2d[:, 2]
    v = points_2d[:, 1] / points_2d[:, 2]

    # 8) Filter valid points: z>0, in image bounds
    z_cam = points_2d[:, 2]
    valid = (z_cam > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)

    u_valid = u[valid]
    v_valid = v[valid]
    lbl_valid = labels[valid]

    # 9) Overlay on image
    for uu, vv, lbl_id in zip(u_valid, v_valid, lbl_valid):
        color = get_color_for_label(lbl_id)
        cv2.circle(img, (int(uu), int(vv)), 2, color, -1)

    # 10) Show or save result
    out_file = 'overlay_00000.png'
    cv2.imwrite(out_file, img)
    print(f"Saved overlay to {out_file}")

if __name__ == "__main__":
    main()
