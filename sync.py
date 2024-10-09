import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, Imu
import message_filters
from cv_bridge import CvBridge
import numpy as np
import tf2_ros
import meshio
import os
from PIL import Image as PILImage
import struct

class DataSynchronizer(Node):

    def __init__(self):
        super().__init__('data_synchronizer')
        self.bridge = CvBridge()

        # Subscribe to /cam_color, /lidar_wp, and /imu
        image_sub = message_filters.Subscriber(self, Image, '/cam_color')
        pointcloud_sub = message_filters.Subscriber(self, PointCloud2, '/lidar_wp')
        imu_sub = message_filters.Subscriber(self, Imu, '/imu')

        # Set up message filter to synchronize topics
        self.ts = message_filters.ApproximateTimeSynchronizer([image_sub, pointcloud_sub, imu_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.synchronized_callback)

        # Setup paths for SemanticKITTI output format
        self.sequence_dir = "path_to_output_folder/sequences/00"
        self.velodyne_dir = os.path.join(self.sequence_dir, "velodyne")
        self.image_2_dir = os.path.join(self.sequence_dir, "image_2")
        self.label_dir = os.path.join(self.sequence_dir, "labels")
        self.calib_file = os.path.join(self.sequence_dir, "calib.txt")
        self.poses_file = os.path.join(self.sequence_dir, "poses.txt")

        # Create necessary directories
        os.makedirs(self.velodyne_dir, exist_ok=True)
        os.makedirs(self.image_2_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)

        self.poses = []

    def synchronized_callback(self, image_msg, pointcloud_msg, imu_msg):
        # Convert image to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

        # Get the timestamp from the message
        message_time = rclpy.time.Time.from_msg(image_msg.header.stamp)
        tolerance = 0.5  # Adjust tolerance as needed

        # Obtain transformation from LIDAR frame to camera frame using TF
        trans = self.lookup_transform_with_tolerance(self.camera_frame, self.lidar_frame, message_time, tolerance)
        if trans is None:
            self.get_logger().warn('Transform not available.')
            return

        # Convert PointCloud2 to PLY file format using meshio
        ply_file_path = os.path.join(self.velodyne_dir, f"{image_msg.header.stamp.sec:06d}.ply")
        self.convert_pointcloud2_to_ply(pointcloud_msg, ply_file_path)

        # Save the point cloud as SemanticKITTI .bin format using meshio
        bin_file_path = os.path.join(self.velodyne_dir, f"{image_msg.header.stamp.sec:06d}.bin")
        self.save_ply_as_bin(ply_file_path, bin_file_path)

        # Save the image as SemanticKITTI .png format
        image_output_path = os.path.join(self.image_2_dir, f"{image_msg.header.stamp.sec:06d}.png")
        pil_image = PILImage.fromarray(cv_image)
        pil_image.save(image_output_path)

        # Process IMU Data to derive pose
        pose = self.get_pose_from_imu(imu_msg)
        self.poses.append(pose)

        # Save poses to poses.txt file
        self.save_poses_to_file()

    def convert_pointcloud2_to_ply(self, pointcloud_msg, ply_file_path):
        # Conversion from ROS2 PointCloud2 to a PLY file using meshio
        try:
            # Assuming the point cloud has fields x, y, z
            points = []
            for point in sensor_msgs_py.read_points(pointcloud_msg, field_names=("x", "y", "z"), skip_nans=True):
                points.append([point[0], point[1], point[2]])

            # Save the points to a PLY file using meshio
            points = np.array(points)
            meshio.write_points_cells(ply_file_path, points, cells={})
        except Exception as e:
            self.get_logger().error(f"Error converting point cloud: {str(e)}")

    def save_ply_as_bin(self, ply_file_path, bin_file_path):
        # Use meshio to read the PLY file and save it as a binary (.bin)
        try:
            mesh = meshio.read(ply_file_path)
            points = mesh.points[:, :3]  # Assuming there are at least x, y, z fields
            intensity = np.zeros((points.shape[0], 1))  # Assuming no intensity data available

            # Combine points and intensity into the desired format
            points_with_intensity = np.hstack((points, intensity)).astype(np.float32)

            # Write the data to a .bin file
            points_with_intensity.tofile(bin_file_path)
        except Exception as e:
            self.get_logger().error(f"Error saving PLY as BIN: {str(e)}")

    def get_pose_from_imu(self, imu_msg):
        # Extract orientation and convert to transformation matrix
        orientation = imu_msg.orientation
        quat = [orientation.x, orientation.y, orientation.z, orientation.w]
        rotation_matrix = self.quaternion_to_rotation_matrix(quat)

        # Assuming velocity integration is handled and position can be obtained
        position = [0.0, 0.0, 0.0]  # Replace with actual position if available

        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rotation_matrix
        pose_matrix[:3, 3] = position

        return pose_matrix[:3, :]  # Return a 3x4 matrix

    def quaternion_to_rotation_matrix(self, quat):
        # Convert quaternion to rotation matrix
        x, y, z, w = quat
        # Normalize the quaternion
        norm = np.linalg.norm(quat)
        x, y, z, w = x / norm, y / norm, z / norm, w / norm

        # Calculate rotation matrix
        rotation_matrix = np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
        ])
        return rotation_matrix

    def save_poses_to_file(self):
        with open(self.poses_file, 'w') as f:
            for pose in self.poses:
                flattened_pose = ' '.join(map(str, pose.flatten()))
                f.write(flattened_pose + '\n')

    def lookup_transform_with_tolerance(self, target_frame, source_frame, time, tolerance):
        # Use TF2 to lookup transform with a given tolerance
        try:
            tf_buffer = tf2_ros.Buffer()
            listener = tf2_ros.TransformListener(tf_buffer)
            trans = tf_buffer.lookup_transform(target_frame, source_frame, time, rclpy.duration.Duration(seconds=tolerance))
            return trans
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return None

def main(args=None):
    rclpy.init(args=args)
    node = DataSynchronizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
