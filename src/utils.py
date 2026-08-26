import math
import torch
import numpy as np
import random
from scipy.spatial.transform import Rotation as R
import subprocess
import time
import os

def create_log_gaussian(mean, log_std, t):
    """Compute element-wise log-probability of sample t under a diagonal Gaussian defined by mean and log_std"""
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable way of computing log( sum_j[ exp(inputs_j) ] )"""
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def sample_random_quaternion(max_angle_deg=90):
    """
    Sample a random quaternion within ±max_angle_deg for roll, pitch, yaw from identity(zero rotation).
    """
    max_angle_rad = math.radians(max_angle_deg)

    # Random roll, pitch, yaw within limits
    roll = random.uniform(-2*max_angle_rad, 2*max_angle_rad)
    pitch = random.uniform(-max_angle_rad, max_angle_rad)
    yaw = random.uniform(-2*max_angle_rad, 2*max_angle_rad)

    # Convert Euler angles (XYZ order) to quaternion
    quat = R.from_euler('xyz', [roll, pitch, yaw]).as_quat()
    # scipy outputs (x, y, z, w)
    return quat


def sample_goal_cartesian(
    x_lim=(-0.5, 0.5), y_lim=(-0.5, 0.5),
    z_lim=(0.9, 1.3)
):
    """
    --------
    Sample a random goal from specified closed intervals
    # Input: intervals x[], y[], z[]
    # Output: random array sampled from within the input intervals [x, y, z]
    --------
    """
    while True:
        rand_x = random.uniform(x_lim[0], x_lim[1])
        rand_y = random.uniform(y_lim[0], y_lim[1])
        rand_z = random.uniform(z_lim[0], z_lim[1])
        distance = math.sqrt(rand_x**2 + rand_y**2 + rand_z**2)
        if distance < z_lim[0] or distance > z_lim[1]:
            continue
        x = rand_x
        y = rand_y
        z = rand_z
        return np.array([x,y,z])


def quaternion_to_euler(x, y, z, w):
    """
    -------
    Convert a quaternion into Euler angles (roll, pitch, yaw) in degrees

    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    -------
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    roll_x = math.degrees(roll_x)
    pitch_y = math.degrees(pitch_y)
    yaw_z = math.degrees(yaw_z)

    return roll_x, pitch_y, yaw_z


def euler_to_quaternion(roll, pitch, yaw):
    """
    -------
    Convert an Euler angle to a quaternion.

    Input
      :param roll: The roll (rotation around x-axis) angle in radians.
      :param pitch: The pitch (rotation around y-axis) angle in radians.
      :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
      :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    -------
    """
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return [qx, qy, qz, qw]

# Gazebo visualization methods:

def set_gazebo_pose(head_pose, world_name="my_world", model_name="skin", timeout=0.05):
    """Method for sending gazebo set_pose -commands for the head model using a service call"""
    # 'timeout'-input is probably set to some default value if it's not set
    x, y, z = head_pose["pos"]
    qx, qy, qz, qw = head_pose["quat"]
    pose_text = f"""
    name: "{model_name}"
    position {{ x: {x:.4f} y: {y:.4f} z: {z:.4f} }}
    orientation {{ x: {qx:.4f} y: {qy:.4f} z: {qz:.4f} w: {qw:.4f} }}
    """
    subprocess.run([
        "gz", "service",
        "-s", f"/world/{world_name}/set_pose",
        "--reqtype", "gz.msgs.Pose",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", str(int(timeout)),
        "--req", pose_text
    ], check=False)  # check=True will raise on non-zero exit; either works depending on desired behavior


def _spawn_marker_model(marker_id: int, pos, orientation, scale=0.02,
                         color_rgba=(1,0,0,1),
                         world_name="my_world"):
    """Spawn a coil model as a visual marker using gz create service."""
    x, y, z = pos
    qx, qy, qz, qw = orientation
    r, g, b, a = color_rgba
    s = scale

    print("Spawning marker at ", pos, orientation)

    # Dynamically find the path to the 'gazebo' folder relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    stl_path = os.path.join(current_dir, 'gazebo', 'coil_xy_short.stl')

    # Format it as a Gazebo-friendly file:// URI
    dynamic_stl_uri = f"file://{stl_path}"

    # Inject the dynamic URI into SDF string
    sdf = f"""<?xml version="1.0" ?>
    <sdf version="1.8">
      <model name="marker_{marker_id}">
        <static>true</static>
        <link name="link">
          <visual name="visual">
          <transparency>0.5</transparency>
            <geometry>
            <mesh>
            <uri>{dynamic_stl_uri}</uri>
            <scale>1 1 1</scale>
            </mesh>
            </geometry>
            <material>
              <ambient>{r} {g} {b} {a}</ambient>
              <diffuse>{r} {g} {b} {a}</diffuse>
              <specular>0.3 0.3 0.3 1.0</specular>
            </material>
          </visual>
        </link>
      </model>
    </sdf>"""

    req = (
        f'name: "marker_{marker_id}" '
        f'pose {{ position {{ x: {x:.4f} y: {y:.4f} z: {z:.4f} }} '
        f'orientation {{ x: {qx:.4f} y: {qy:.4f} z: {qz:.4f} w: {qw:.4f} }} }} '
        f'sdf: "{sdf.replace(chr(10), " ").replace(chr(34), chr(92)+chr(34))}"'
    )

    subprocess.run([
        "gz", "service",
        "-s", f"/world/{world_name}/create",
        "--reqtype", "gz.msgs.EntityFactory",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "5",
        "--req", req
    ], check=False)


def _delete_marker_model(marker_id: int, world_name="my_world"):
    """Delete a previously spawned marker model by name."""
    req = f'name: "marker_{marker_id}" type: MODEL'
    subprocess.run([
        "gz", "service",
        "-s", f"/world/{world_name}/remove",
        "--reqtype", "gz.msgs.Entity",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "1",
        "--req", req
    ], check=False)


def visualize_goal_in_gazebo(goal_pos_world, orientation, normal_world=None,
                              marker_id_base=100, world_name="my_world"):
    """
    Visualize goal using spawned SDF models.
    Call _delete_marker_model() for each id before re-spawning on next goal.
    """
    # Delete previous markers first
    for i in range(1):
        _delete_marker_model(marker_id_base + i, world_name)

    time.sleep(0.05)

    # Red sphere at goal contact point
    _spawn_marker_model(marker_id_base, goal_pos_world, orientation,
                        scale=0.02, color_rgba=(1, 0, 0, 1), world_name=world_name)

    # Cyan sphere offset along normal (shows normal direction)
    """if normal_world is not None:
        n = np.asarray(normal_world)
        n = n / (np.linalg.norm(n) + 1e-12)
        normal_tip = goal_pos_world + 0.05 * n
        _spawn_marker_model(marker_id_base + 1, normal_tip,
                            scale=0.012, color_rgba=(0, 1, 1, 1), world_name=world_name)
        # Small sphere midway so it looks like an arrow shaft
        midpoint = goal_pos_world + 0.025 * n
        _spawn_marker_model(marker_id_base + 2, midpoint,
                            scale=0.008, color_rgba=(0, 1, 1, 0.7), world_name=world_name)
"""

