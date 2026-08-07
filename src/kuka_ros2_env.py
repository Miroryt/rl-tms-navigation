import os
import rclpy
import time

from rclpy.node import Node
from std_srvs.srv import Empty
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import gymnasium as gym
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import math

from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from std_msgs.msg import Float64MultiArray

from utils import sample_random_quaternion, sample_goal_cartesian, visualize_goal_in_gazebo, set_gazebo_pose
from Iiwa14FK import Iiwa14FK
from action_space import DistanceAdaptiveActionSpace


from scipy.spatial.transform import Rotation as R

from mesh_utils import MeshUtils
import numpy as np
import pyvista as pv

from utils import quaternion_to_euler
from head_motion import HeadMotion

from ikpy.chain import Chain
from ikpy.link import URDFLink

import json

class KukaRos2Env(gym.Env):
    def __init__(self,

                 # CORRECT THIS 'my_world.sdf' PATH FOR YOUR MACHINE

                 sdf_path='/home/user/rl-tms-navigation/src/gazebo/my_world.sdf',

                 head_model_name='skin',
                 training=False,
                 goal='random',
                 collision=False,
                 plots=False,
                 run_itself=False,
                 action_EE_coordinates=False):
        super().__init__()
        self.training = training
        self.goal = goal
        self.collision = collision
        self.plots = plots
        self.run_itself = run_itself
        self.action_EE_coordinates = action_EE_coordinates

        # Where to store the accumulated data
        self.data_file = os.path.expanduser("~/kuka_env_data.json")


        # CORRECT THIS PATH TO 'iiwa14.urdf' MATCHING THE LOCATION ON YOUR DEVICE
        iiwa14_kinematic_chain = "/home/user/rl-tms-navigation/src/gazebo/iiwa14_kinematic.urdf"

        self.fk_model = Iiwa14FK(
            urdf_path=iiwa14_kinematic_chain,  # match the URDF launched in Gazebo
            base_link='lbr_link_0',
            ee_link='lbr_link_ee'
        )
        self.node = None  # No ROS2 node when training

        """Mesh Utilities Object"""
        self.mesh_utils = MeshUtils(sdf_path, model_name=head_model_name)

        #Mesh in robot frame from mesh_utils.py
        self.head_mesh_pv = pv.wrap(self.mesh_utils.head_mesh_r)
        self.head_mesh_pv_copy = pv.wrap(self.mesh_utils.head_mesh_r)

        if not self.training:
            """Initialize ROS2 client library (mandatory for creating nodes or interacting with ROS2)"""
            rclpy.init()
            """Create a Node called kuka_rl_env"""
            self.node = rclpy.create_node('kuka_rl_env')

            """Quality of Service (QoS)
            A QoS profile defines a set of policies, including durability, reliability, queue depth and sample history storage"""
            """
            @param depth: Keep the last x messages
            @param reliability: 
                Best effort = attempt to deliver samples, but may lose them if the network is not robust.
                Reliable = guarantee that samples are delivered, may retry multiple times.
            @param history:         
                Keep last: only store up to N samples, configurable via the queue depth option.
                Keep all: store all samples, subject to the configured resource limits of the DDS vendor.
            """
            qos = QoSProfile(
                depth=5,
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
            )

            """Create a Subscriber subscribing to the robot's joint state topic"""
            self.joint_sub = self.node.create_subscription(
                JointState,
                '/lbr/joint_states',
                self.joint_callback,
                qos
            )

            """Create a subscriber for robot's joint torques"""
            self.torque_sub = self.node.create_subscription(
                JointState,
                '/lbr/joint_states',
                self.torque_callback,
                qos
            )

            """Create an Action client to send trajectory goals to the robot. Sends a FollowJointTrajectory goal to the controller"""
            self.action_client = ActionClient(
                self.node,
                FollowJointTrajectory,
                '/lbr/joint_trajectory_controller/follow_joint_trajectory'
            )

        """Store the latest 7-DOF joint position array:"""
        self.current_joint_state = np.zeros(7)
        """For training, create another position array:"""
        self.sim_joint_state = np.zeros(7)
        """Flag to indicate that a first valid joint state has been received"""
        self.received_first_state = False

        """Joint limits converted to radians"""
        self.joint_limits = {
            "lower": np.radians([-170, -120, -170, -120, -170, -120, -175]),
            "upper": np.radians([170, 120, 170, 120, 170, 120, 175])
        }

        """Define the robot's start position in joint space"""
        """Joint state angles are in radians"""
        #Pre-defined home-position
        self.home_position = np.array([
            -0.15,
            -0.70,
            0.13,
            -1.83,
            0.18,
            0.27,
            -0.03
        ])
        print("COBOT HOME POSITION SET TO: ", self.home_position)

        if self.action_EE_coordinates == True:
            """Make EE coordinates and orientation as action"""
            """In gym action spaces we can define different dimensions for different inputs"""
            """It is wise to make spacial movement inputs smaller (meters) and euler angle inputs larger (radians)"""
            self.action_space = gym.spaces.Box(
                low=np.array([-1, -1, -1, -1, -1, -1], dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1], dtype=np.float32),
                dtype=np.float32
            )

            self.adaptive_action = DistanceAdaptiveActionSpace() # Initialize distance adaptive action space
        else:
            """A small delta range, e.g. -0.3 to 0.3 rad per joint"""
            self.action_space = gym.spaces.Box(low=-0.2, high=0.2, shape=(7,), dtype=np.float32)


        if self.action_EE_coordinates == True:

            # I couldn't get Chain creation to work from the urdf file, so I added it manually:
            self.chain = Chain(name="iiwa14_7dof", links=[

                URDFLink(
                    name="A1",
                    origin_translation=[0, 0, 0.1475],
                    origin_orientation=[0, 0, 0],
                    rotation=[0, 0, 1],
                    joint_type="revolute",
                    bounds=(np.radians(-170), np.radians(170))
                ),

                URDFLink(
                    name="A2",
                    origin_translation=[0, -0.01, 0.2125],
                    origin_orientation=[0, 0, 0],
                    rotation=[0, 1, 0],
                    joint_type="revolute",
                    bounds=(np.radians(-120), np.radians(120))
                ),

                URDFLink(
                    name="A3",
                    origin_translation=[0, 0.01, 0.228],
                    origin_orientation=[0, 0, 0],
                    rotation=[0, 0, 1],
                    joint_type="revolute",
                    bounds=(np.radians(-170), np.radians(170))
                ),

                URDFLink(
                    name="A4",
                    origin_translation=[0, 0.0105, 0.192],
                    origin_orientation=[0, 0, 0],
                    rotation=[0, -1, 0],
                    joint_type="revolute",
                    bounds=(np.radians(-120), np.radians(120))
                ),

                URDFLink(
                    name="A5",
                    origin_translation=[0, -0.0105, 0.2075],
                    origin_orientation=[0, 0, 0],
                    rotation=[0, 0, 1],
                    joint_type="revolute",
                    bounds=(np.radians(-170), np.radians(170))
                ),

                URDFLink(
                    name="A6",
                    origin_translation=[0, -0.0707, 0.1925],
                    origin_orientation=[0, 0, 0],
                    rotation=[0, 1, 0],
                    joint_type="revolute",
                    bounds=(np.radians(-120), np.radians(120))
                ),

                URDFLink(
                    name="A7",
                    origin_translation=[0, 0.0707, 0.091],
                    origin_orientation=[0, 0, 0],
                    rotation=[0, 0, 1],
                    joint_type="revolute",
                    bounds=(np.radians(-175), np.radians(175))
                ),
            ])
            print("Chain created: ", self.chain)

        """observation = np.concatenate([
            joint_positions,              # current state
            ee_position,
            ee_orientation,               # current pose
            target_position,              # task goal
            target_orientation,
            distance_to_goal
        ])"""
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7 + 3 + 4 + 3 + 4 + 1,), dtype=np.float32)

        self.goal_position = np.zeros(3) #x,y,z coordinates
        self.goal_orientation = np.array([0, 0, 0, 1]) #quaternion
        self.random_position = np.zeros(3) #x,y,z coordinates

        """------ For moving targets -------"""
        self.goal_pos_array = []
        self.goal_workspace = {'x': (-0.5, 0.5), 'y': (-0.5, 0.5), 'z': (0.9, 1.3)}
        self.control_dt = 0.5
        self._moving_base_goal = sample_goal_cartesian()
        self.moving_base_quat = sample_random_quaternion()
        self._moving_idx = 0
        self._moving_start_time = None
        self._sine_params = {
            'amp': np.array([0.01, 0, 0.01]),
            'freq': np.array([0.1, 0, 0.1]),
            'phase': np.random.rand(3) * 2 * np.pi
        }
        self.goal_local = None
        self.normal_local = None
        self.goal_pose = None
        self.head_motion = HeadMotion(training=training)
        self.t0 = time.time()
        self.initial_sample = None
        self.coil_angle = None
        "Coil_angle does not change as the head moves, only the tangent frame rotates with head !"

        #Previous end effectors position and time values for calculating the speed of the EE:
        self.prev_ee_position = None
        self.prev_ee_velocity = None
        self.prev_joint_state = None
        self.prev_time = None

        self.prev_head_pose = None
        self.p_w = None

        """Limit each episode"""
        self.max_episode_steps = 100
        self.step_counter = 0
        self.ema_velocity = np.zeros(3)

        self.num_violations = 0
        self.pos_err_field = 0
        self.angle_err_field = 0
        self.collision_field = 0
        self.n_r = None #Head surface normal
        self.ee_normal = None #End effector normal
        self.n_r_local = None
        self.prev_pos_err = None
        self.prev_angle_err = None


        """----- GRAPHS -----"""
        # Create new empty arrays for graphs in reset()
        #Per-episode arrays
        self.array_times = []  # list of t values
        self.goal_positions = []
        self.ee_positions = []
        self.distances = []
        self.array_work = []
        self.array_angle_error = []
        self.all_episode_collisions = []
        self.collision_count = 0
        self.joint_state_history = []
        self.joint_time_history = []
        self.joint_matrix = 0
        self.joint_times = []
        self.collision_speed = []
        self.actions = []
        self.head_position = []
        self.head_rotation = []

        # Across-episodes storage
        self.all_ee_positions = [] # list of arrays, one per episode
        self.all_distances = []
        self.all_work = []
        self.all_angle_errors = []
        self.all_joint_matrices = []
        self.all_joint_times = []
        self.all_goal_positions = []
        self.all_collision_speeds = []
        self.all_actions = []
        self.all_head_positions = []
        self.all_head_rotations = []


        """------------------"""

        """----- REWARD LOGGING -----"""
        self.position_reward = []
        self.orientation_reward = []
        self.joint_penalty = []
        self.work_penalty = []
        self.speed_penalty = []
        self.collision_penalty = []
        self.exponential_bonus = []
        self.normal_penalty = []

        self.episode_start_time = None
        self.last_joint_update = 0
        self.has_sent_trajectory = False # for static head motion planner loop

        # Load previous saved data if it exists
        self._load_saved_data()

    def _load_saved_data(self):
        """Load previous episode data from JSON file (if it exists)."""
        if os.path.exists(self.data_file):
            with open(self.data_file, "r") as f:
                data = json.load(f)

            # Convert back to numpy arrays
            self.all_distances = [np.array(d) for d in data.get("distances", [])]
            self.all_work = [np.array(w) for w in data.get("work", [])]
            self.all_angle_errors = [np.array(a) for a in data.get("angles", [])]
            self.all_episode_collisions = data.get("collisions", [])
            self.all_joint_matrices = [np.array(m) for m in data.get("joint_matrices", [])]
            self.all_joint_times = [np.array(t) for t in data.get("joint_times", [])]
            self.all_goal_positions = [np.array(g) for g in data.get("goal_positions", [])]
            self.all_ee_positions = [np.array(e) for e in data.get("ee_positions", [])]
            self.all_collision_speeds = [np.array(e) for e in data.get("collision_speeds", [])]
            self.all_actions = [np.array(aa) for aa in data.get("actions", [])]
            self.all_head_positions = [np.array(hh) for hh in data.get("head_positions", [])]
            self.all_head_rotations = [np.array(rr) for rr in data.get("head_rotations", [])]

            # If current session lacks joint_times/matrix, use the last saved ones
            if len(self.all_joint_times) > 0 and (self.joint_times is None):
                self.joint_times = self.all_joint_times[-1].copy()
            if len(self.all_joint_matrices) > 0 and (self.joint_matrix is None):
                self.joint_matrix = self.all_joint_matrices[-1].copy()

            print(f"[INFO] Loaded {len(self.all_distances)} episode(s) from {self.data_file}")
        else:
            print("[INFO] No saved data found — starting fresh.")

    def _save_data(self):
        """Save all episode data to JSON file (numpy -> lists)."""
        data = {
            "distances": [d.tolist() for d in self.all_distances],
            "work": [w.tolist() for w in self.all_work],
            "angles": [a.tolist() for a in self.all_angle_errors],
            "collisions": list(self.all_episode_collisions),
            "joint_matrices": [m.tolist() for m in self.all_joint_matrices],
            "joint_times": [t.tolist() for t in self.all_joint_times], #self.joint_times
            "goal_positions": [g.tolist() for g in self.all_goal_positions],
            "ee_positions": [e.tolist() for e in self.all_ee_positions],
            "collision_speeds": [c.tolist() for c in self.all_collision_speeds],
            "actions": [aa.tolist() for aa in self.all_actions],
            "head_positions": [hh.tolist() for hh in self.all_head_positions],
            "head_rotations": [rr.tolist() for rr in self.all_head_rotations]
        }
        # ensure directory exists
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        #This rewrites the json file every time it's called, thus data arrays cannot be cleared between episodes
        with open(self.data_file, "w") as f:
            json.dump(data, f)
        print(f"[INFO] Saved episode data to {self.data_file}")
        print(f"[INFO] Data file now consists of {len(self.all_distances)} episode(s)")


    def append_episode_data(self, distances, work, angles, collisions, joint_matrix, joint_times, all_goal_positions, collision_speeds, actions, head_positions, head_rotations):
        """
        This is called at the end of an episode to append and persist the episode's arrays.
        distances, work, angles -> 1D arrays (timesteps)
        collisions -> real number per episode
        joint_matrix -> 2D array shape (timesteps, n_joints)
        joint_times -> 1D array of time stamps (timesteps)
        """
        #These store ALL episode data arrays and they cannot be cleared !
        self.all_distances.append(np.array(distances))
        self.all_work.append(np.array(work))
        self.all_angle_errors.append(np.array(angles))
        self.all_episode_collisions.append(int(collisions))
        self.all_joint_matrices.append(np.array(joint_matrix))
        self.all_joint_times.append(np.array(joint_times))
        self.all_goal_positions.append(np.array(all_goal_positions))
        self.all_ee_positions.append(np.array(self.ee_positions))
        self.all_collision_speeds.append(np.array(collision_speeds))
        self.all_actions.append(np.array(actions))
        self.all_head_positions.append(np.array(head_positions))
        self.all_head_rotations.append(np.array(head_rotations))

        # update current episode memory
        self.joint_matrix = np.array(joint_matrix)
        self.joint_times = np.array(joint_times)

        # persist immediately
        self._save_data()

    def work_cost(self, q, q_prev):
        """
        --------
        Estimate 'work' as weighted sum of joint angular displacements.
        Weights = cumulative link masses from proximal to distal.
        # Input: current joint state, previous joint state
        # Output: work estimate
        --------
        """
        # Link masses extracted from iiwa14 URDF
        link_masses = np.array(self.fk_model.get_link_masses()[:7])  # 7 links

        # Compute cumulative masses from distal → proximal
        # Example: [m1+...+m7, m2+...+m7, ..., m7]
        cumulative_masses = np.array([np.sum(link_masses[i:]) for i in range(len(link_masses))])

        # Angular displacements
        dq = np.abs(q - q_prev)  # rad (absolute to avoid cancellations)

        # Weighted sum of displacements
        work = np.sum(cumulative_masses * dq)

        return work

    #We can also calculate work from torques with a ROS2 node
    #But this is not practical for training, since running ROS2 is computationally heavy for training
    def work_from_torques(self, q, q_prev):
        """Method to calculate work from torques with a ROS2 node"""
        dq = q - q_prev  # difference in joint states (rad)
        work_joints = self.current_joint_torque * dq
        total_work = np.sum(work_joints)
        return total_work, work_joints

    def wait_for_action_server(self):
        """Wait until the action server is available. Throws an error message if it doesn't response in 100 seconds"""
        if not self.action_client.wait_for_server(timeout_sec=100):
            self.node.get_logger().error("Action server not available!")
            raise RuntimeError("Action server not available")

    def joint_callback(self, msg):
        """Updates the current joint state when a new JointState message arrives"""
        self.current_joint_state = np.array(msg.position)
        self.received_first_state = True
        self.last_joint_update = time.time()

    def torque_callback(self, msg):
        """Updates the current joint torque when a new JointState message arrives"""
        self.current_joint_torque = np.array(msg.effort)
        self.received_first_state = True

    # Define a ForwardPositionController method to send joint targets to the cobot
    # Not used
    def send_joint_positions(self, q_desired):
        msg = Float64MultiArray()
        msg.data = q_desired.tolist()
        self._forward_position_pub.publish(msg)


    def send_joint_trajectory(self, target_pos: np.ndarray, duration_sec: float = 0.2):
        """
        -------
        Send a trajectory to the robot via ROS2 node. The FollowJointTrajectory Action client is slower than using
        LBRJointPositionCommandController, but the Gazebo launch
        (ros2 launch lbr_bringup gazebo.launch.py) doesn't support/launch that topic (?)
        # Input: target position, movement duration
        -------
        """

        """Create a goal and specify joint names (lbr_A1... lbr_A7)"""
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = [f'lbr_A{i+1}' for i in range(7)]

        """Set target joint positions:"""
        point = JointTrajectoryPoint()
        point.positions = target_pos.tolist()
        """Set movement duration"""
        point.time_from_start.sec = int(duration_sec)
        goal_msg.trajectory.points.append(point)

        """Ensure action server is available:"""
        self.wait_for_action_server()
        """Sends the goal asynchronously and blocks until it is accepted"""
        future = self.action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self.node, future)
        goal_handle = future.result()

        """Check if the goal was accepted:"""
        if not goal_handle.accepted:
            self.node.get_logger().error("Goal rejected by action server")
            raise RuntimeError("Trajectory goal rejected")

        """Wait for motion to finish:"""
        self.node.get_logger().info("Goal accepted, waiting for result...")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future)
        self.node.get_logger().info("Motion complete.")


    def reset(self, seed=None, options=None):
        """
        --------
        Resets the environment
        # Output: observation
        --------
        """
        self.t0 = time.time() #episode start time
        self.has_sent_trajectory = False

        # Print previous episode info and acquire ee_pos and ee_rot
        ee_position, ee_rotation = self.prev_ep_info_print()
        # Reset all graph logs for a fresh episode
        self.reset_graphs()

        if hasattr(self.mesh_utils, 'x_axis_local'):
            self.mesh_utils.x_axis_local = None

        # ---- OU-state reset at episode boundary ----
        self.head_motion.reset()

        if self.training: # No ROS2 preparations needed

            # If you want to set simulated starting position to random:
            self.random_position = np.random.uniform(
                low=self.joint_limits["lower"],
                high=self.joint_limits["upper"]
            ).astype(np.float32)
            self.sim_joint_state = self.home_position
            back_off = False

            # --- Simulated starting position by backing off from previous final position ---
            if self.n_r is not None and back_off == True:
                # Backoff distance
                backoff_dist = 0.15

                # Retreat position (robot frame)
                print("Backing off from: ", ee_position, "Direction (vector): ", self.n_r)
                retreat_pos = ee_position + backoff_dist * self.n_r
                retreat_quat = ee_rotation

                # --- Build target_T ---
                R_target = R.from_quat(retreat_quat).as_matrix()

                target_T = np.eye(4)
                target_T[:3, :3] = R_target
                target_T[:3, 3] = retreat_pos

                # --- Solve IK ---
                q_target = self.chain.inverse_kinematics_frame(
                    target=target_T,
                    initial_position=self.current_joint_state,
                    orientation_mode="all"
                )
                self.sim_joint_state = q_target
                self.current_joint_state = self.sim_joint_state

            else:
                #Fixed simulation starting position for cobot:
                #self.sim_joint_state = self.home_position.copy()
                self.current_joint_state = self.sim_joint_state

            ee_position, ee_rotation = self.fk_model.fk(self.sim_joint_state)
            """Normalization"""
            ee_rotation = ee_rotation / np.linalg.norm(ee_rotation)

            if self.goal == 'moving_random':
                """Random goal quaternion that is tilting with sine wave"""
                self.moving_base_quat = sample_random_quaternion()
                """Random goal position that is moving with sine wave"""
                self._moving_base_goal = sample_goal_cartesian()
                self.goal_position = self._moving_base_goal.copy()
                self._moving_idx = 0
                self._moving_start_time = time.time()

            if self.goal == 'moving_head': #This goal is moved in step() -function
                """Get a random goal on the head mesh"""
                self.goal_position, self.goal_orientation, self.n_r, self.n_r_local, self.p_w, self.coil_angle = self.mesh_utils.sample_goal(offset=0.081)
                self.moving_base_quat = self.goal_orientation.copy()
                self._moving_base_goal = self.goal_position.copy()
                self.initial_sample = self.p_w  # in WORLD coordinates

        if not self.training: # ROS2 preparations are needed
            """Wait until first valid joint state is received"""
            while not self.received_first_state:
                self.node.get_logger().info('Waiting for first joint state...')
                rclpy.spin_once(self.node, timeout_sec=0.2)
            """Reset the flag and send the robot to home position:"""
            self.received_first_state = False

            if self.run_itself == True:
                self.send_joint_trajectory(self.home_position, duration_sec=1.0)
            ee_position, ee_rotation = self.fk_model.fk(self.current_joint_state.copy())
            """Normalization"""
            ee_rotation = ee_rotation / np.linalg.norm(ee_rotation)

            """Featuring 2 different target practises"""
            """--------------------------------------"""
            if self.goal == 'moving_random':
                self.goal_orientation = sample_random_quaternion()
                """Random goal position that is moving with sine wave"""
                self._moving_base_goal = sample_goal_cartesian()
                self.goal_position = self._moving_base_goal.copy()
                self._moving_idx = 0
                self._moving_start_time = time.time()
                print("Set goal position to: ", self.goal_position)
                print("Set goal orientation to: ", self.goal_orientation)
            if self.goal == 'moving_head':
                # Reset the cached local coordinates from the previous episode.
                self.mesh_utils.sample_local = None
                self.mesh_utils.normal_local = None
                if hasattr(self.mesh_utils, 'x_axis_local'):
                    self.mesh_utils.x_axis_local = None
                """Get a random goal on a static head mesh"""
                self.goal_local, self.goal_orientation, self.n_r, self.n_r_local, self.p_w, self.coil_angle = self.mesh_utils.sample_goal(offset=0.081)
                self.normal_local = self.n_r

                while not self.received_first_state:
                    self.node.get_logger().info('Waiting for first joint state...')
                    rclpy.spin_once(self.node, timeout_sec=0.2)
                """Reset the flag and send the robot to home position:"""
                self.received_first_state = False
                if self.run_itself == True:
                    self.send_joint_trajectory(self.home_position, duration_sec=1.0)
                else:
                    # Backoff distance
                    backoff_dist = 0.15

                    # Retreat position (robot frame)
                    print("Backing off from: ", ee_position, "Direction (vector): ", self.n_r)
                    retreat_pos = ee_position + backoff_dist * self.n_r
                    retreat_quat = ee_rotation

                    # --- Build target_T ---
                    R_target = R.from_quat(retreat_quat).as_matrix()

                    target_T = np.eye(4)
                    target_T[:3, :3] = R_target
                    target_T[:3, 3] = retreat_pos

                    # --- Solve IK ---
                    q_target = self.chain.inverse_kinematics_frame(
                        target=target_T,
                        initial_position=self.current_joint_state,
                        orientation_mode="all"
                    )
                    # --- Execute ---
                    self.send_joint_trajectory(q_target, duration_sec=1.0)

                ee_position, ee_rotation = self.fk_model.fk(self.sim_joint_state)
                """Normalization"""
                ee_rotation = ee_rotation / np.linalg.norm(ee_rotation)

                if self.run_itself == False:

                    self.user_input_control()  # Ask the user for a goal or sample a random goal on the head surface

                # Skip asking for user input and sample a random goal on the head:
                else:
                    """Get a random goal on the head mesh"""
                    self.goal_position, self.goal_orientation, self.n_r, self.n_r_local, self.p_w, self.coil_angle = self.mesh_utils.sample_goal(
                        offset=0.081)
                    self.initial_sample = self.p_w  # in WORLD coordinates
                    """Normalization"""
                    self.goal_orientation = self.goal_orientation / np.linalg.norm(self.goal_orientation)
                    print(">>> Using random head goal:", self.goal_position, self.goal_orientation)

        """Normalization"""
        self.goal_orientation = self.goal_orientation / np.linalg.norm(self.goal_orientation)


        distance_to_goal = np.linalg.norm(ee_position - self.goal_position)
        distance_to_goal = np.array([distance_to_goal], dtype=np.float32) #Convert to vector for numpy

        """Observation. MUST BE IN THE SAME ORDER AS IN THE OBSERVATION SPACE TO AVOID CONFUSION FOR THE AGENT"""
        obs = np.concatenate([
            self.sim_joint_state if self.training else self.current_joint_state,
            ee_position,
            ee_rotation,
            self.goal_position,
            self.goal_orientation,
            distance_to_goal
        ])

        # at episode end:
        if self.training is False:
            temp_flag = np.any(self.joint_times) and np.any(self.distances)
            if temp_flag:
                self.append_episode_data(
                    distances=self.distances,
                    work=self.array_work,
                    angles=self.array_angle_error,
                    collisions=self.collision_count,
                    joint_matrix=self.joint_matrix,  # shape (T, 7)
                    joint_times=self.joint_times,  # shape (T,)
                    all_goal_positions=self.goal_positions,
                    collision_speeds=self.collision_speed,
                    actions=self.actions,
                    head_positions=self.head_position,
                    head_rotations=self.head_rotation
                )

                # Empty per-episode data arrays
                self.distances = []
                self.array_work = []
                self.array_angle_error = []
                self.joint_times = []
                self.goal_positions = []
                self.ee_positions = []
                self.collision_speed = []
                self.actions = []
                self.head_position = []
                self.head_rotation = []

        """Reset num of collisions"""
        self.collision_field = 0
        self.collision_count = 0

        return obs, {}


    def step(self, action):
        """
        --------
        Step function, handles sending actions, updating simulation, computing the reward etc. by calling different functions
        # Input: Action (SAC output)
        # Output: observation, reward, terminated, truncated, info
        --------
        """
        if self.action_EE_coordinates == True:
            if self.training:
                ee_position, ee_rotation = self.fk_model.fk(self.sim_joint_state)
            else:
                joint_state = self.current_joint_state.copy()
                ee_position, ee_rotation = self.fk_model.fk(joint_state)


            distance = np.linalg.norm(ee_position - self.goal_position)
            action = self.adaptive_action.scale_action(action, distance)

        # step counter
        self.step_counter += 1


        # If using EE coordinates as action, compute target EE transform
        if self.action_EE_coordinates == True:
            current_T = self.get_current_ee_transform(self.current_joint_state)
            delta_T = self.build_delta_transform(action)
            target_T = current_T @ delta_T

            # IKPy: use inverse_kinematics_frame
            q_target = self.chain.inverse_kinematics_frame(
                target=target_T,
                initial_position=self.current_joint_state,
                orientation_mode="all"  # "all" if you want to track orientation
            )
        else:
            q_target = None

        if self.training:
            """Skip sending actions to the joint_trajectory_client"""
            if self.action_EE_coordinates == True: #Action is EE coordinates + orientation
                self.sim_joint_state = q_target
                self.current_joint_state = self.sim_joint_state

            else: #Action is joint state update
                self.sim_joint_state += action # Agent's action is added to current joint states

            ee_position, ee_rotation = self.fk_model.fk(self.sim_joint_state)
            """Normalization"""
            ee_rotation = ee_rotation / np.linalg.norm(ee_rotation)
            """Get end-effector normal:"""
            # quaternion to rotation matrix
            R_ee = R.from_quat(ee_rotation).as_matrix()

            # EE z-axis in world/robot frame (depends on convention)
            self.ee_normal = R_ee[:, 2]  # third column = local +Z axis in global frame

            self.goal_orientation = self.goal_orientation / np.linalg.norm(self.goal_orientation)
            joint_state = self.sim_joint_state.copy()

            """"""
            if self.goal == 'moving_random':

                self.goal_position, self.goal_orientation = self.generate_random_moving_goal()

            if self.goal == 'moving_head':
                # Get current head pose
                #t = time.time() - self.t0
                t_head = self.head_motion.advance_time()
                head_pose = self.head_motion.get_pose(t_head)
                # Move the sampled goal point on the head according to head movement
                # Update goal position/orientation
                self.goal_position, self.goal_orientation = self.mesh_utils.transform_goal_by_head_pose(
                    self.goal_position,
                    self.prev_head_pose,
                    head_pose,
                    self.initial_sample,  # only used the very first time per episode
                    self.n_r,
                    self.n_r_local,
                    offset=0.081,
                    coil_angle=self.coil_angle
                )
                self.prev_head_pose = head_pose

        else: #Evaluation
            self.actions.append(action)  # Log action

            joint_state = self.current_joint_state.copy()
            ee_position, ee_rotation = self.fk_model.fk(joint_state)
            """Normalization"""
            ee_rotation = ee_rotation / np.linalg.norm(ee_rotation)

            if self.goal == 'moving_random':
                self.goal_position, self.goal_orientation = self.generate_random_moving_goal()

            # Get current head pose
            t_head = self.head_motion.advance_time()
            if self.goal == 'moving_head':
                head_pose = self.head_motion.get_pose(t_head)  # x, y, z, = head_pose["pos"]
                self.head_position.append(head_pose["pos"])
                x, y, z, w = head_pose["quat"]
                roll, pitch, yaw = quaternion_to_euler(x, y, z, w)
                yaw = yaw + 90 # Normalize to 0, the head is spawned 90 degrees counterclockwise in gazebo, so remove that to log the change in angle
                self.head_rotation.append([roll, pitch, yaw])
                #self.head_rotation = head_pos_euler
                set_gazebo_pose(head_pose, world_name="my_world", model_name="skin", timeout=0.05)
                # Update head trimesh location for collision detection
                self.update_head_mesh(head_pose)

                self.goal_position, self.goal_orientation = self.mesh_utils.transform_goal_by_head_pose(
                    self.goal_position,
                    self.prev_head_pose,
                    head_pose,
                    self.initial_sample,
                    self.n_r,
                    self.n_r_local,
                    offset=0.081,  # 0.09 used previously in training
                    coil_angle=self.coil_angle
                )
                # print("-----Goal position: ", self.goal_position)
                # print("-----Goal quaternion: ", self.goal_orientation)
                self.prev_head_pose = head_pose


            # Update graphs
            self.update_graphs(ee_position, ee_rotation, self.goal)

            # SEND ACTION TO COBOT
            self.send_action(q_target, action) #q_target = target joint state (7-vector), 'action' is for actions that are already joint angles (self.action_EE_coordinates == False)
            # -----------------------

            #print("ACTION: ", action)

            if self.step_counter == 2 and self.goal=='moving_head': #Ensure the head position is correct before visualizing the goal in gazebo
                # p_w and n_r_local are already in world frame — visualize those directly
                goal_world = self.mesh_utils.robot_to_world_points(self.goal_position)
                visualize_goal_in_gazebo(
                    goal_pos_world=self.mesh_utils.position_goal_to_marker(self.goal_position, self.n_r_local, offset=0.08),  # convert the actual goal
                    orientation=self.mesh_utils.orientation_goal_to_marker(self.goal_orientation),
                    normal_world=self.n_r_local,
                    marker_id_base=100,
                )

        distance_to_goal = np.linalg.norm(ee_position - self.goal_position)
        distance_to_goal = np.array([distance_to_goal], dtype=np.float32)  # Convert to vector for numpy

        obs = np.concatenate([joint_state, ee_position, ee_rotation, self.goal_position, self.goal_orientation, distance_to_goal])
        #print("obs:", obs)
        #print("Goal Position: ", self.goal_position, "Goal orientation:", self.goal_orientation)


        """----- Compute Reward -----"""
        truncated = False
        reward, truncated = self.compute_reward(ee_position, ee_rotation, joint_state)

        if self.goal!='moving_head':
            """----- Update previous parameters -----"""
            self.prev_ee_position = ee_position
            self.prev_joint_state = joint_state.copy()
            self.prev_time = time.time()

        """Episode ends if step_counter reaches max"""
        terminated = False
        if self.step_counter >= self.max_episode_steps:
            truncated = True

        info = {
            "num_violations": self.num_violations,
            "pos_err": self.pos_err_field,
            "angle_err": self.angle_err_field,
            "collision": self.collision_field if self.collision else 0
        }

        """Gym-style output: """
        return obs, reward, terminated, truncated, info

    def compute_reward(self, ee_position, ee_rotation, joint_state):
        """
        ---------------------------
        #----- Reward Function -----
        # Input: EE position, EE orientation, current joint state
        # Output: Reward for that step
        ---------------------------
        """

        truncated = False
        reward = 0

        """----- Position and orientation errors -----"""

        pos_err = np.linalg.norm(ee_position - self.goal_position)
        dot = np.clip(np.dot(ee_rotation, self.goal_orientation), -1.0, 1.0)
        angle_err = 2 * np.arccos(np.abs(dot))  # arccos of -1 - 1

        proximity_weight = np.clip(pos_err / 0.05, 0.0, 1.0)  # fades to 0 inside 5cm

        if self.prev_pos_err is not None:
            reward += 5.0 * proximity_weight * (self.prev_pos_err - pos_err)
            self.position_reward.append(5 * proximity_weight * (self.prev_pos_err - pos_err))

        if self.prev_angle_err is not None:
            reward += 3.0 * proximity_weight * (self.prev_angle_err - angle_err)
            self.orientation_reward.append(3 * proximity_weight * (self.prev_angle_err - angle_err))


        self.prev_pos_err = pos_err
        self.prev_angle_err = angle_err

        """----- Broad linear guidance -----"""
        reward -= 0.3 * pos_err
        reward -= 0.3 * angle_err

        """----- Multi-tier exponential reward -----"""
        pos_term = (
                2.0 * np.exp(-8 * pos_err) +  # broad   (scale ~12.5 cm)
                2.0 * np.exp(-30 * pos_err) +  # medium  (scale ~3.3 cm) #4
                10.0 * np.exp(-200 * pos_err)  # fine    (scale ~5 mm)
        )

        angle_term = (
                2.0 * np.exp(-8 * angle_err) +  # broad   (scale ~7.2 deg)
                4.0 * np.exp(-10 * angle_err) +  # medium  (scale ~5.7 deg)
                10.0 * np.exp(-200 * angle_err)  # fine    (scale ~1.9 deg)
        )

        reward += pos_term
        reward += angle_term

        self.position_reward.append(pos_term)
        self.orientation_reward.append(angle_term)

        # Clear target for the agent
        POS_THRESHOLD = 0.002  # 2mm
        ANGLE_THRESHOLD = 0.087  # ~5 degrees

        if pos_err < POS_THRESHOLD and angle_err < ANGLE_THRESHOLD:
            reward += 20.0
            print("PERFECT")
            self.position_reward.append(20.0)

        """Update class fields for tensorboard"""
        self.pos_err_field = pos_err
        self.angle_err_field = angle_err


        """----- End Effector velocity penalty -----"""
        dt = self.control_dt  # fixed physical control period, matches head_motion (0.5s)
        speed = None
        ema_alpha = 0.1  # EMA coefficient
        if self.prev_ee_position is not None and self.prev_time is not None:
            ee_velocity = (ee_position - self.prev_ee_position) / dt
            self.ema_velocity = ema_alpha * ee_velocity + (1 - ema_alpha) * self.ema_velocity
            speed = np.clip(np.linalg.norm(self.ema_velocity), 0, 20)

            self.prev_ee_velocity = ee_velocity

        """----- Work cost of action -----"""
        if self.prev_joint_state is not None:
            work_estimate = self.work_cost(joint_state, self.prev_joint_state)
            reward -= 0.01 * work_estimate
            self.work_penalty.append(-0.01 * work_estimate)
        else:
            self.work_penalty.append(0)

        """-----------------------------------------"""


        """------ Collision detection between EE and head mesh -------"""
        """End effector mesh approximates as a small sphere for collision detection:"""

        """Collision detection takes significant computing power and slows down training !"""
        """------ Collision detection between EE and head mesh -------"""
        if pos_err < 0.10:
            if self.collision:
                collided = False

                # --- SPHERE GOALS ---
                if self.goal in ['random', 'moving_random']:
                    # Simple Euclidean Distance Check (No Mesh needed)
                    dist = np.linalg.norm(ee_position - self.goal_position)
                    # Collision if distance < (EE_radius + Goal_radius)
                    min_dist = dist - 0.081
                    if min_dist < 0.08:
                        collided = True
                    else:
                        collided = False

                # --- MESH GOALS (Optimized Trimesh) ---
                elif self.goal in ['head', 'moving_head']:
                    head_pos = None
                    head_quat = None
                    ee_radius = 0.08

                    if self.goal == 'moving_head':
                        t_head = self.head_motion.advance_time()
                        head_pose = self.head_motion.get_pose(t_head)
                        if self.training == False:
                            set_gazebo_pose(head_pose, world_name="my_world", model_name="skin", timeout=0.05)
                            self.update_head_mesh(head_pose)
                        head_pos = head_pose["pos"]
                        head_quat = head_pose["quat"]

                    dist_to_surface, collided = self.mesh_utils.compute_collision(
                        ee_pos_robot=ee_position,
                        head_pos_world=head_pos,
                        head_quat_world=head_quat,
                        ee_radius=ee_radius
                    )

                if collided:
                    reward -= 5
                    self.collision_penalty.append(-5)
                    if speed is not None:
                        reward -= 1 * speed
                        self.collision_penalty.append(-1 * speed)


                    print(
                        f"--- COLLIDED at {speed if speed is not None else 0:.3f} m/s ---")
                    self.collision_field += 1
                    self.collision_count += 1
                    self.collision_speed.append(speed)

                    truncated = False
                else:
                    self.collision_penalty.append(0)

        if self.training == False:
            print("----------------")
            print("Distance to goal: ", pos_err)
            print("Orientation error: ", angle_err)
            print("reward: ", reward)
            print("----------------")
        if self.training == True:
            print("Distance: ", pos_err, " Orientation: ", angle_err)

        return reward, truncated

    def generate_random_moving_goal(self):
        """
        ------------
        # Method that outputs a goal that is shifted with sin-function (to simulate movement)
        # Output: cartesian coordinates [x, y, z], orientation quaternion [q1, q2, q3, q4]
        ------------
        """
        idx = self._moving_idx
        t = idx * self.control_dt
        noise = np.random.normal(0, 0.005, 1)
        """Create a sine wave with parameters"""
        sine = self._sine_params['amp'] * np.sin(
            2.0 * np.pi * self._sine_params['freq'] * t) + noise
        """Create a new goal with stationary goal + sine"""
        new_goal = self._moving_base_goal + sine
        """Make sure the goal is in action space bounds"""
        """numpy.clip() = Values outside the interval will be capped to the interval edges"""
        new_goal[0] = np.clip(new_goal[0], *self.goal_workspace['x'])
        new_goal[1] = np.clip(new_goal[1], *self.goal_workspace['y'])
        new_goal[2] = np.clip(new_goal[2], *self.goal_workspace['z'])
        goal_position = new_goal
        """Create a new goal orientation with initial orientation + sine"""
        pitch = 0.4 * math.sin(0.1 * t)  # radians
        R_base = R.from_quat(self.moving_base_quat)
        R_pitch = R.from_euler('y', pitch)  # rotation about y-axis
        R_new = R_pitch * R_base  # apply pitch tilt on top of base orientation

        goal_orientation = R_new.as_quat()
        self._moving_idx += 1

        return goal_position, goal_orientation

    def update_graphs(self, ee_position, ee_rotation, goal='moving_head'):
        """
        ---------------
        # Input: EE position, EE orientation, goal
        ---------------
        """

        t = time.time() - self.t0  # elapsed time since episode start
        self.joint_state_history.append(self.current_joint_state.copy())
        self.joint_time_history.append(t)

        if goal == 'moving_head':  # !=
            self.joint_matrix = np.array(self.joint_state_history)  # shape: (steps, num_joints)
            self.joint_times = np.array(self.joint_time_history)  # shape: (steps,)

            # Record head's position and ee distance at each time point
            self.array_times.append(t)
            self.goal_positions.append(self.goal_position)
            self.ee_positions.append(ee_position)
            self.distances.append(np.linalg.norm(ee_position - self.goal_position))

            # Record EE orientation's z-axis and compute alignment error with surface normal
            R_ee = R.from_quat(ee_rotation).as_matrix()  # transform quaternion into matrix
            z_ee = R_ee[:, 2]  # 3rd column = z-axis
            R_goal = R.from_quat(self.goal_orientation).as_matrix()
            n_robot_t = R_goal[:, 2]  # EE should align with this

            cos_angle = np.dot(z_ee, n_robot_t) / (np.linalg.norm(z_ee) * np.linalg.norm(n_robot_t))
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

            self.array_angle_error.append(angle_deg)

            if self.prev_joint_state is not None:
                try:
                    work_estimate = self.work_cost(self.current_joint_state, self.prev_joint_state)
                    work_estimate = work_estimate  # Joules
                    self.array_work.append(work_estimate)
                except TypeError:
                    work_estimate = 0
                    self.array_work.append(work_estimate)
            else:
                self.array_work.append(0)  # Fill first place so array length matches with time array

    def send_action(self, q_target, action):
        """
        -----------
        # This method calls self.send_joint_trajectory and waits until the robot has reached the desired position before continuing
        # Input: Target joint angles (depending on if SAC is outputting joint angles or EE pos + orientation)
        -----------
        """

        """Apply the action and send it to the cobot:"""
        if self.action_EE_coordinates == True:
            target_position = q_target  # SEND THE IK JOINT ANGLES TO THE ROBOT
            # self.current_joint_state = q_target
        else:
            target_position = self.current_joint_state + action * 1  # SEND THE CURRENT JOINT STATE + AGENT'S ACTION TO THE ROBOT

        # ---------------------
        self.send_joint_trajectory(target_position)  # !!!
        # ---------------------

        # print(action)
        joint_state = self.current_joint_state.copy()
        ee_position, ee_rotation = self.fk_model.fk(joint_state)
        """Normalization"""
        ee_rotation = ee_rotation / np.linalg.norm(ee_rotation)
        self.goal_orientation = self.goal_orientation / np.linalg.norm(self.goal_orientation)
        """Get end-effector normal:"""
        # quaternion to rotation matrix
        R_ee = R.from_quat(ee_rotation).as_matrix()

        # EE z-axis in world/robot frame (depends on convention)
        self.ee_normal = R_ee[:, 2]  # third column = local +Z axis in global frame

        # Wait until the joint state is close enough to target
        timeout = 3.0  # seconds
        start_time = time.time()
        #rclpy.spin_once(self.node, timeout_sec=0.5)
        # If the joint space euclidian distance from current joint state and target one is greater than x, we need to wait
        while np.linalg.norm(self.current_joint_state - target_position) > 0.01:  # 0.01 previously
            rclpy.spin_once(self.node, timeout_sec=0.01)
            if time.time() - start_time > timeout:
                print("[WARN] Motion timeout.")
                break

    def prev_ep_info_print(self):
        """ PREVIOUS EPISODE INFO PRINT """

        print("--------------------------------------------------------")
        print("The goal pos.:", self.goal_position, "The goal orient.:", self.goal_orientation)
        if self.coil_angle is not None:
            print("Coil angle (deg): ", np.rad2deg(self.coil_angle))
        if not self.training:
            ee_position, ee_rotation = self.fk_model.fk(self.current_joint_state.copy())
        if self.training:
            ee_position, ee_rotation = self.fk_model.fk(self.sim_joint_state)
        """Normalization"""
        ee_rotation = ee_rotation / np.linalg.norm(ee_rotation)
        or_goal_euler = quaternion_to_euler(self.goal_orientation[0], self.goal_orientation[1],
                                            self.goal_orientation[2], self.goal_orientation[3])
        dot = np.clip(np.dot(ee_rotation, self.goal_orientation), -1.0, 1.0)
        angle_err = 2 * np.arccos(np.abs(dot))
        print("Goal or in euler degrees: ", or_goal_euler)
        print("Pos.: ", ee_position, "Orient.: ", ee_rotation)
        print("Norm of distance difference: ", np.linalg.norm(ee_position - self.goal_position))
        print("Ang. dist. (rad): ", angle_err)
        print("Number of joint violations: ", self.num_violations)
        print("Number of collisions: ", self.collision_count)
        print("--------------------------------------------------------")

        print("REWARD OVERVIEW:")
        print("Position: ", np.sum(self.position_reward))
        print("Orientation: ", np.sum(self.orientation_reward))
        print("Work: ", np.sum(self.work_penalty))
        print("Speed: ", np.sum(self.speed_penalty))
        print("Joint limits: ", np.sum(self.joint_penalty))
        print("Collisions: ", np.sum(self.collision_penalty))
        # print("Normal penalty: ", np.sum(self.normal_penalty))

        print("--------------------------------------------------------")

        print("---------------------------------")
        print("------- EPISODE STARTING --------")
        print("---------------------------------")

        return ee_position, ee_rotation

    def reset_graphs(self):
        """ Method for resetting episode-wide graphs """

        """Reset step counter"""
        self.step_counter = 0
        """Reset num of joint violations"""
        self.num_violations = 0
        """Reset joint state history"""
        self.joint_state_history = []  # list to store joint states
        self.joint_time_history = []
        """Reset reward logging"""
        self.position_reward = []
        self.orientation_reward = []
        self.joint_penalty = []
        self.work_penalty = []
        self.speed_penalty = []
        self.collision_penalty = []
        self.exponential_bonus = []
        self.normal_penalty = []
        # self.joint_times = [0]  # Reset time axis from plots at the start of every episode
        self.prev_head_pose = None  # Reset previous head pose
        """Reset motion planner trajectory and index"""
        self._planner_traj = None
        self._planner_idx = 0

        # Reset the cached local coordinates from the previous episode.
        # This forces transform_goal_by_head_pose to re-initialize
        # using the new initial_sample.
        self.mesh_utils.sample_local = None
        self.mesh_utils.normal_local = None

    def user_input_control(self):
        """ Method for asking the user to pick the target during simulation """

        #Ask the user for next coordinates, or a random one on the head:
        # ---- Interactive part ----
        user_in = input(
            "Enter goal as 'x y z qx qy qz qw' or press enter for random (moving) head goal: "
        )
        if user_in.strip().lower() == "yes" or user_in.strip() == "":

            """Get a random goal on the head mesh"""
            self.goal_position, self.goal_orientation, self.n_r, self.n_r_local, self.p_w, self.coil_angle = self.mesh_utils.sample_goal(
                offset=0.081)
            self.initial_sample = self.p_w  # in WORLD coordinates
            """Normalization"""
            self.goal_orientation = self.goal_orientation / np.linalg.norm(self.goal_orientation)
            print(">>> Using random head goal:", self.goal_position, self.goal_orientation)
        else:
            try:
                vals = list(map(float, user_in.strip().split()))
                if len(vals) != 7:
                    raise ValueError
                pos = np.array(vals[:3], dtype=np.float32)
                quat = np.array(vals[3:], dtype=np.float32)
                quat /= np.linalg.norm(quat)  # normalize quaternion
                self.goal_position, self.goal_orientation = pos, quat
                self.n_r = None
                print(">>> Using manual goal:", self.goal_position, self.goal_orientation)
            except ValueError:
                print("Invalid input! Falling back to random head goal.")
                self.goal_position, self.goal_orientation, self.n_r, self.n_r_local, self.p_w, self.coil_angle = self.mesh_utils.sample_goal(
                    offset=0.081)
                self.initial_sample = self.p_w  # in WORLD coordinates
                """Normalization"""
                self.goal_orientation = self.goal_orientation / np.linalg.norm(self.goal_orientation)

    def update_head_mesh(self, head_pose):
        """ Update head mesh pose in Python to match Gazebo pose """

        pos_w = np.array(head_pose["pos"]) #position in world frame
        pos_r = self.mesh_utils.world_to_robot_points(pos_w) #position in robot frame

        # Compute centroid of robot-frame mesh
        centroid_r = self.mesh_utils.head_mesh_r.centroid

        # Compute translation to align centroid with Gazebo position
        translation = pos_r - centroid_r

        # Apply translation
        transformed_vertices = self.mesh_utils.head_mesh_r.vertices + translation
        self.head_mesh_pv_copy.points = transformed_vertices

        #print("POS WORLD: ", pos_w, "POS ROBOT: ", pos_r, "CENTROID ROBOT: ", centroid_r)

    def wait_for_fresh_joint_state(self, timeout=0.05):
        """ Call rclpy.spin_once until we get a new joint state message from joint_callback """
        t0 = time.time()
        last = self.last_joint_update
        while self.last_joint_update == last:
            rclpy.spin_once(self.node, timeout_sec=0.001)
            if time.time() - t0 > timeout:
                return False
        return True

    def close(self):
        if getattr(self, 'training', False):
            # No ROS node in training node
            return
        if hasattr(self, 'node'):
            self.node.destroy_node()
        try:
            rclpy.shutdown()
        except rclpy._rclpy.RCLError:
            pass  # already shutdown

    def get_current_ee_transform(self, current_joint_state):
        """
        --------------
        Returns the transformation matrix of the forward kinematics
        # Input: joint state (7x1)
        # Output: the transformation matrix (4x4)
        --------------
        """
        current_joint_state = np.array(current_joint_state, dtype=float).flatten()
        ee_frame = self.chain.forward_kinematics(current_joint_state)  # 4x4
        return ee_frame

    def build_delta_transform(self, action):
        """
        # ------------
        # Input: action (EE position + orientation, this method is only called when self.action_EE_coordinates == True)
        # Output:
        # ------------
        """
        dx, dy, dz, droll, dpitch, dyaw = action

        # Translation delta
        T_delta = np.eye(4)
        T_delta[:3, 3] = [dx, dy, dz]

        # Rotation delta
        rot = R.from_euler('xyz', [droll, dpitch, dyaw]).as_matrix()
        T_delta[:3, :3] = rot

        return T_delta

