import time
import subprocess
import math
import colorednoise as cn
import numpy as np
from scipy.fft import fft
from mesh_utils import MeshUtils
from scipy.spatial.transform import Rotation as R

"""head_motion.py sends gazebo commands to set new pose to the head (move the head)
with /world/my_world/set_pose service"""

class HeadMotion:
    def __init__(self, training=True, control_dt=0.5):
        """
        HeadMotion-class updated to better match real-life head movement
        -----------------------
        OptiTrack measurements:
        X lateral:    peak-to-peak ~3.5 mm,  std ~0.41 mm
        Y depth:      peak-to-peak ~7.2 mm,  std ~0.84 mm
        Z vertical:   peak-to-peak ~5.7 mm,  std ~1.15 mm
        Roll:         peak-to-peak ~2.56 deg, std ~0.32 deg
        Pitch:        peak-to-peak ~4.72 deg, std ~0.44 deg
        Yaw:          peak-to-peak ~3.32 deg, std ~0.31 deg
        """

        self.training = training

        self.control_dt = control_dt  # fixed physical control period, same in train & eval
        self._sim_time = 0.0

        # Base position (world coords, meters)
        self.base_x = 1.0
        self.base_y = 0.0
        self.base_z = 0.5

        # Position amplitudes (meters)
        # Divide measured peak-to-peak amplitude by two to get sinusoidal amplitude

        """ For example x:
        slow + fast = peak_to_peak / 2
        ------------------------------
        0.00120 + 0.00055 = 0.00175
         0.00175 * 2 = 0.0035 = 3.5 mm
        """
        self.pos_amps = {
            "x": (0.00120, 0.00055),
            "y": (0.00250, 0.00110),
            "z": (0.00200, 0.00085),
        }

        # Rotation amplitudes (radians)
        self.rot_amps = {
            "pitch": (0.0280, 0.0130),
            "roll": (0.0155, 0.0070),
            "yaw": (0.0200, 0.0090)
        }

        self.pos_noise_std = {
            "x": 0.00035,  # 0.35 mm
            "y": 0.00072,  # 0.72 mm
            "z": 0.00057,  # 0.57 mm
        }

        self.rot_noise_std = {
            "pitch": math.radians(0.008), #0.47 degrees
            "roll": math.radians(0.0045), #0.26 degrees
            "yaw": math.radians(0.006), #0.33 degrees
        }

        # OU noise state — one scalar per axis, initialized to 0
        self._ou_state = {k: 0.0 for k in ["x", "y", "z", "pitch", "roll", "yaw"]}
        self._ou_tau = 0.5  # seconds — physiological noise correlation time
        self._prev_t = None  # tracks last get_pose(t) call; dt derived from this

        # Random per-axis phase offsets so axes don't all peak simultaneously
        rng = np.random.default_rng()
        self.phase = {k: rng.uniform(0, 2 * math.pi)
                      for k in ["x", "y", "z", "pitch", "roll", "yaw",
                                "x_fast", "y_fast", "z_fast",
                                "pitch_fast", "roll_fast", "yaw_fast"]}

        self.freq_slow = 2 * math.pi * 0.03  # Most common frequency in Optitrack recording
        self.freq_fast = 2 * math.pi * 0.06  # 2nd most common frequency

    def advance_time(self):
        """Advance simulated time by exactly one control period,
        regardless of how long this Python call actually took."""
        self._sim_time += self.control_dt
        return self._sim_time

    def _ou_step(self, key, noise_std, dt):
        """
        One OU step: dx = -(x/tau)*dt + sigma*sqrt(2/tau)*sqrt(dt)*N(0,1)
        Stationary variance = noise_std^2
        Step-to-step std = 2*noise_std*sqrt(dt/tau) << noise_std for tau >> dt.
        """
        theta = 1.0 / self._ou_tau
        sigma = noise_std * math.sqrt(2.0 / self._ou_tau)
        dW = math.sqrt(dt) * np.random.normal()
        self._ou_state[key] += -theta * self._ou_state[key] * dt + sigma * dW
        return self._ou_state[key]

    def _signal(self, amp_slow, amp_fast, phase_slow, phase_fast, noise_std, t, dt, noise_key):
        slow = amp_slow * math.sin(self.freq_slow * t + phase_slow)
        fast = amp_fast * math.sin(self.freq_fast * t + phase_fast)
        noise = self._ou_step(noise_key, noise_std, dt)
        return slow + fast + noise

    def get_pose(self, t, movement=True):
        """
        ---------
        Outputs transformed head coordinates and orientation
        # Output:
                     "pos":  np.array([x, y, z]),
                     "quat": R_total.as_quat()  # (x, y, z, w)

        Time t is always advanced by 0.5 seconds, regardless of if we are training or evaluating
        --------
        """
        # one RL step always corresponds to 0.5 s of head motion, whether that step takes 50 microseconds in training (only python)
        # or 0.5 real seconds waiting on the actual robot in evaluation (running gazebo and ros2)
        dt = self.control_dt

        if not movement: # Don't move the head
            yaw_base = -1.57
            R_yaw = R.from_euler('z', yaw_base)
            return {
                "pos":  np.array([self.base_x, self.base_y, self.base_z]),
                "quat": R_yaw.as_quat()
            }

        # ----- OU -------
        # Position
        x = self.base_x + self._signal(
            *self.pos_amps["x"], self.phase["x"], self.phase["x_fast"],
            self.pos_noise_std["x"], t, dt, "x")
        y = self.base_y + self._signal(
            *self.pos_amps["y"], self.phase["y"], self.phase["y_fast"],
            self.pos_noise_std["y"], t, dt, "y")
        z = self.base_z + self._signal(
            *self.pos_amps["z"], self.phase["z"], self.phase["z_fast"],
            self.pos_noise_std["z"], t, dt, "z")
        # Rotation

        yaw_base = -1.57  # head facing sideways
        pitch = self._signal(
            *self.rot_amps["pitch"], self.phase["pitch"], self.phase["pitch_fast"],
            self.rot_noise_std["pitch"], t, dt, "pitch")
        roll = self._signal(
            *self.rot_amps["roll"], self.phase["roll"], self.phase["roll_fast"],
            self.rot_noise_std["roll"], t, dt, "roll")
        yaw_delta = self._signal(
            *self.rot_amps["yaw"], self.phase["yaw"], self.phase["yaw_fast"],
            self.rot_noise_std["yaw"], t, dt, "yaw")

        # --------------------------------------------------------------------
        # Compose: base yaw -> pitch around local X -> roll around local Y
        R_yaw   = R.from_euler('z', yaw_base + yaw_delta)
        local_x = R_yaw.apply([1, 0, 0])
        local_y = R_yaw.apply([0, 1, 0])
        R_pitch = R.from_rotvec(pitch * local_x)
        R_roll  = R.from_rotvec(roll  * local_y)
        R_total = R_roll * R_pitch * R_yaw

        return {
            "pos":  np.array([x, y, z]),
            "quat": R_total.as_quat()  # (x, y, z, w)
        }

    def reset(self):
        """Call at episode start."""
        self._sim_time = 0.0
        self._ou_state = {k: 0.0 for k in self._ou_state}
        self._prev_t = None
        rng = np.random.default_rng()
        self.phase = {k: rng.uniform(0, 2 * math.pi) for k in self.phase}

