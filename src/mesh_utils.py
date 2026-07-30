"""Imports for loading head model mesh"""

from geometry_msgs.msg import PoseStamped
import os
import xml.etree.ElementTree as ET
import trimesh
from urllib.parse import urlparse, unquote
from scipy.spatial.transform import Rotation as R
from trimesh.sample import sample_surface
import numpy as np
import pyvista as pv

"""Robot's coordinate axis and gazebo's world axis are different, such that for
   x_robot = -z_world
   y_robot = y_world
   z_robot = x_world
"""


class MeshUtils:
    def __init__(self, sdf_path, model_name="skin", assume_mm=False):
        self.sdf_path = sdf_path
        self.model_name = model_name
        self.assume_mm = assume_mm

        # Get File Info
        pose, mesh_path, scale = self._get_model_pose_and_mesh()
        if not os.path.isabs(mesh_path):
            mesh_path = os.path.abspath(mesh_path)

        self.sdf_pose_array = pose
        self.sdf_scale = scale
        self.mesh_path = mesh_path

        self.coil_angle = None # Initialize coil_angle

        # COORDINATE TRANSFORMS (World <-> Robot)
        # rotation: world -> robot
        self.R_wr = np.array([[0, 0, -1],
                              [0, 1, 0],
                              [1, 0, 0]], dtype=float)
        self.t_wr = np.array([1.0, 0.0, 0.0], float)

        # rotation: robot -> world
        self.R_rw = self.R_wr.T
        # translation: robot -> world (p_w = R_rw @ p_r - R_rw @ t_wr)
        self.t_rw = -self.R_rw @ self.t_wr

        # MESH LOADING

        # Canonical Mesh (For Collision):
        # Loads mesh at (0,0,0) with no rotation/translation, only scale.
        # This is the "Local Space" mesh used for fast queries.
        self.collision_mesh = trimesh.load(mesh_path, force='mesh')
        self.collision_mesh.apply_scale(scale)
        if assume_mm:
            self.collision_mesh.apply_scale(0.001)

        # Build the BVH Tree once
        self.proximity_query = trimesh.proximity.ProximityQuery(self.collision_mesh)

        # World/Robot Mesh:
        mesh_world = self._mesh_in_world(mesh_path, pose, scale, assume_mm=self.assume_mm)
        self.head_mesh = mesh_world  # World Frame

        # Transform vertices to Robot Frame for visualization
        Vw = mesh_world.vertices
        Vr = (Vw @ self.R_wr.T) + self.t_wr
        self.head_mesh_r = trimesh.Trimesh(vertices=Vr, faces=mesh_world.faces, process=False)

    # ------------------------------------------------------------------------
    # COLLISION CHECK
    # ------------------------------------------------------------------------
    def compute_collision(self, ee_pos_robot, head_pos_world=None, head_quat_world=None, ee_radius=0.08):
        """
        Calculates distance using Inverse Transformations + BVH.
        Returns: (min_distance, is_collision_bool)
        """

        # Convert EE from Robot Frame -> World Frame
        #ee_pos_world = self.robot_to_world_points(ee_pos_robot)
        ee_pos_world = self.R_rw @ ee_pos_robot - self.R_rw @ self.t_wr

        # Determine Head Transform (World -> Head Local)
        if head_pos_world is None:
            # If static, use the initial SDF pose
            x, y, z, roll, pitch, yaw = self.sdf_pose_array
            head_pos = np.array([x, y, z])
            head_rot = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
        else:
            # If moving, use the provided live pose
            head_pos = np.array(head_pos_world)
            head_rot = R.from_quat(head_quat_world).as_matrix()  # Assumes [x,y,z,w]

        # Inverse Transform: Map EE point into the Head's "Canonical" Local Space
        # P_local = R_head_inv * (P_world - T_head)
        p_rel = ee_pos_world - head_pos
        p_local = p_rel @ head_rot  # Transpose of orthogonal matrix is inverse

        # Query the Static BVH Tree
        # signed_distance: + is outside, - is inside
        sd = self.proximity_query.signed_distance([p_local])[0]

        # Check against EE sphere radius
        # If we are 0.05m outside, and radius is 0.08m, we collided.
        # So collision if (distance_to_surface < radius)
        distance_to_surface = -sd
        is_collision = distance_to_surface < ee_radius

        return distance_to_surface, is_collision


    # ------------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------------

    @staticmethod
    def _file_uri_to_path(uri: str) -> str:
        if not uri.startswith('file:'):
            return uri
        u = urlparse(uri)
        path = os.path.normpath(unquote(u.path))
        if u.netloc:
            path = f"/{u.netloc}{path}"
        if not path.startswith('/'):
            path = '/' + path
        return path

    def _get_model_pose_and_mesh(self):
        """A function that finds head mesh (name: skin) pose, path and scale from gazebo world 'my_world.sdf'"""
        root = ET.parse(self.sdf_path).getroot()
        model = root.find(f"./world/model[@name='{self.model_name}']")
        if model is None:
            raise RuntimeError(f"Model '{self.model_name}' not found in {self.sdf_path}")

        pose_el = model.find("pose")
        pose = np.fromstring(pose_el.text, sep=' ') if pose_el is not None else np.zeros(6)
        #print("POSE FROM GET_POSE: ", pose)

        uri_el = model.find(".//link/visual/geometry/mesh/uri")
        if uri_el is None:
            raise RuntimeError("No <uri> under link/visual/geometry/mesh")
        mesh_path = self._file_uri_to_path(uri_el.text.strip())

        scale_el = model.find(".//link/visual/geometry/mesh/scale")
        scale = np.fromstring(scale_el.text, sep=' ') if scale_el is not None else np.ones(3)

        return pose, mesh_path, scale

    def _mesh_in_world(self, mesh_path, pose, scale, assume_mm=False):
        """
        -------
        A method that updates the mesh pose according to inputs
        # Input: mesh path, pose [x, y, z], scale, assume mm
        # Output: transformed mesh
        """
        mesh = trimesh.load(mesh_path, force='mesh')
        mesh.apply_scale(scale)
        if assume_mm:
            mesh.apply_scale(0.001)

        x, y, z, roll, pitch, yaw = pose
        T = np.eye(4)
        T[:3, :3] = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
        T[:3, 3] = [x, y, z]
        #print("T-matrix:", T)
        mesh.apply_transform(T)
        return mesh

    def world_to_robot_points(self, P):
        """Applies world -> robot transform to points (supports (3,) or (N,3) row vectors)"""
        P = np.asarray(P, dtype=float)
        if P.ndim == 1:
            return self.R_wr @ P + self.t_wr
        return (P @ self.R_wr.T) + self.t_wr

    # translation: robot -> world (p_w = R_rw @ p_r - R_rw @ t_wr)
    def robot_to_world_points(self, P):
        """Applies robot -> world transform to points (supports (3,) or (N,3) row vectors)"""
        P = np.asarray(P, dtype=float)
        if P.ndim == 1:
            return self.R_rw @ P + self.t_rw
        return (P @ self.R_rw.T) + self.t_rw

    def position_goal_to_marker(self, P, normal_vector, offset):
        """
        This method transforms goal_position from kuka_ros2_env to correct position (world coordinates + offset transform) for goal visualization.
        P: (x, y, z) + normal_vector: (x, y, z) + offset: (x) original
        returns: (x, y, z) transformed
        """
        P = np.asarray(P, dtype=float)
        if P.ndim == 1:
            return self.R_rw @ P + self.t_rw - offset * normal_vector
        return (P @ self.R_rw.T) + self.t_rw

    def orientation_goal_to_marker(self, quat_r: np.ndarray) -> np.ndarray:
        """
        This method transforms goal_orientation from kuka_ros2_env to correct orientation for goal visualization.
        quat_r: (x, y, z, w) original
        returns: (x, y, z, w) transformed
        """
        R_goal_robot = R.from_quat(quat_r).as_matrix()
        R_goal_world = self.R_rw @ R_goal_robot

        # 90° correction around z-axis (surface normal) to align
        # visual coil model frame with EE frame definition
        R_correction = R.from_euler('x', -90, degrees=True).as_matrix()
        R_goal_world = R_goal_world @ R_correction

        return R.from_matrix(R_goal_world).as_quat()

    def world_to_robot_normals(self, N):
        """Applies transform to normals (rotation only; supports (3,) or (N,3))"""
        N = np.asarray(N, dtype=float)
        if N.ndim == 1:
            Nr = self.R_wr @ N
            return Nr / (np.linalg.norm(Nr) + 1e-12)
        Nr = N @ self.R_wr.T
        Nr /= (np.linalg.norm(Nr, axis=1, keepdims=True) + 1e-12)
        return Nr


    def sample_goal(self, offset=0, coil_angle: float = None):
        """
            Returns approach_point, quat, n_r, n_r_local, p_w, coil_angle

            coil_angle: rotation (radians) of the coil around the surface normal.
                        If None, a random angle in [15°, 75°] is sampled.
        """
        # constrain the area, where samples are taken from
        z_threshold = 0.58
        x_threshold_high = 1.05  # 1.1
        x_threshold_low = 0.90  # 0.9
        y_threshold_low = -0.2  # -0.1
        y_threshold_high = 0.2  # 0.1

        while True:
            try:
                points_w, face_idx = sample_surface(self.head_mesh, 50)
            except RuntimeError:
                # Handle case where mesh is empty or can't be sampled
                raise RuntimeError("Could not sample from head mesh.")

            mask = (points_w[:, 2] > z_threshold) & \
                   (points_w[:, 0] < x_threshold_high) & \
                   (points_w[:, 0] > x_threshold_low) & \
                   (points_w[:, 1] < y_threshold_high) & \
                   (points_w[:, 1] > y_threshold_low)

            #Filter the points and their corresponding face indices
            valid_points = points_w[mask]
            valid_faces = face_idx[mask]

            # When we find a valid point, get out of the loop
            if len(valid_points) != 0:
                print("Valid sample point found!")
                break

        #Randomly select one valid point/face
        idx = np.random.randint(len(valid_points))
        p_w = valid_points[idx]
        face_index_on_original_mesh = valid_faces[idx]

        #Get the normal from the *original* mesh
        n_w = self.head_mesh.face_normals[face_index_on_original_mesh]

        # Sample random coil angle if not provided
        # it is important to notice that tangent direction depends on where you are on the head !
        if coil_angle is None:
            # ±30° around the 45° (=[15°, 75°]) posterior-anterior direction covers clinical use in MOTOR CORTEX HOTSPOT FINDING
            # It is also recommended clinical practice to iterate different coil angles to find the strongest response.
            # This serves as a reason to have a continuous space of coil angles in agent training
            coil_angle = np.deg2rad(45) + np.random.uniform(-np.deg2rad(30), np.deg2rad(30))

        # convert to robot frame (points: R·p + t; normals: R·n)
        p_r = self.world_to_robot_points(p_w)
        n_r = self.world_to_robot_normals(n_w)

        n_r_local = n_w

        # Normalize surface normal vector (on the target coordinates)
        n_r = n_r / (np.linalg.norm(n_r) + 1e-12)

        # move offset distance (m) along the surface normal
        approach_point = p_r + offset * n_r

        # print("------Sampled world point:", p_w)
        # print("------Transformed robot point:", p_r)
        # print("nr: ", n_r)
        # print("offset * n_r: ", offset * n_r)
        # print("approach point: ", approach_point)

        # build EE orientation in ROBOT frame: +Z of EE points INTO surface
        z_axis = -n_r  # normalized already, change sign INTO surface
        world_x = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(z_axis, world_x)) > 0.95:
            world_x = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(world_x, z_axis);
        x_axis /= np.linalg.norm(x_axis)  # normalization
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)  # normalization

        Rm = np.column_stack((x_axis, y_axis, z_axis))  # already in robot frame
        quat = R.from_matrix(Rm).as_quat()  # (x,y,z,w)

        return approach_point, quat, n_r, n_r_local, p_w, coil_angle


    def transform_goal_by_head_pose(self,
                                    prev_goal_pos,
                                    prev_head_pose,
                                    head_pose,
                                    initial_sample,
                                    n_r,  # (unused)
                                    n_r_local,  # This is the initial world normal (n_w_sdf)
                                    offset,
                                    coil_angle: float):
        """
        Transforms the goal based on the head's current pose.
        Calculates the goal from scratch to prevent drift and handle rotation.

        coil_angle is stored at first call and re-applied consistently on
        subsequent calls, so the target doesn't drift as the head moves.
        """

        # --- Initialization & Main Calculation ---
        if self.sample_local is None:
            # --- This is the FIRST run ---
            if prev_head_pose is not None:
                raise RuntimeError("In first call, prev_head_pose needs to be None")

            try:
                t_sdf = self.sdf_pose_array[0:3]
                R_sdf = R.from_euler('xyz', self.sdf_pose_array[3:6]).as_matrix()
            except AttributeError:
                raise RuntimeError("self.sdf_pose_array not set in __init__!")

            # Store the local (head-frame) sample point and normal
            self.sample_local = R_sdf.T @ (initial_sample - t_sdf)
            self.normal_local = R_sdf.T @ n_r_local
            self.offset = offset
            self.coil_angle = coil_angle

            # Set the current world-frame values (for this first run)
            approach_point = prev_goal_pos
            n_world = n_r_local  # Use the initial world normal

        else:
            # --- This is a SUBSEQUENT run ---

            # Get current head pose
            t_head_curr = head_pose["pos"]
            R_head_curr = R.from_quat(head_pose["quat"]).as_matrix()

            # Transform local sample point to CURRENT world frame
            p_w_surface = R_head_curr @ self.sample_local + t_head_curr

            # Transform local normal to CURRENT world frame
            n_world = R_head_curr @ self.normal_local
            n_world /= (np.linalg.norm(n_world) + 1e-12)  # Normalize

            # Calculate the offset goal point in the CURRENT world frame
            approach_point_world = p_w_surface + self.offset * n_world

            #approach_point_world = p_w_surface
            approach_point = self.R_wr @ approach_point_world + self.t_wr

        # --- Transform WORLD goal to ROBOT frame ---
        n_robot = self.R_wr @ n_world
        n_robot /= (np.linalg.norm(n_robot) + 1e-12)  # normalize

        # --- Orientation Calculation ---
        _, orientation = self._orientation_from_normal(n_robot, self.coil_angle)


        return approach_point, orientation

    @staticmethod
    def _orientation_from_normal(n_r: np.ndarray, coil_angle: float) -> np.ndarray:
        """
        Build a 3×3 rotation matrix for the EE such that:
          - EE +Z points INTO the surface  (-n_r)
          - EE X/Y are rotated by coil_angle around that axis
        Returns R_goal (3×3), quaternion (x,y,z,w)
        """
        z_axis = -n_r / (np.linalg.norm(n_r) + 1e-12)

        # Build a stable, fixed reference tangent frame first
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(z_axis, ref)) > 0.95:
            ref = np.array([0.0, 1.0, 0.0])

        x_base = np.cross(ref, z_axis)
        x_base /= np.linalg.norm(x_base)
        y_base = np.cross(z_axis, x_base)
        y_base /= np.linalg.norm(y_base)

        # Apply coil rotation around the normal axis
        c, s = np.cos(coil_angle), np.sin(coil_angle)
        x_axis = c * x_base + s * y_base
        y_axis = -s * x_base + c * y_base

        R_goal = np.column_stack((x_axis, y_axis, z_axis))
        quat = R.from_matrix(R_goal).as_quat()  # (x,y,z,w)
        return R_goal, quat


