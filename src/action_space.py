import numpy as np

class DistanceAdaptiveActionSpace:
    """
    Scales per-step Cartesian action bounds based on EE distance to goal.
    Far from goal  -> larger bound (faster travel, makes training faster).
    Near goal      -> smaller bound (more fine-tuned movements near the head.

    Uses smoothstep interpolation rather than a hard cutoff at `near_distance`,
    so the bound itself doesn't introduce a discontinuity at the
    boundary
    """

    def __init__(
        self,
        near_bound=np.array([0.005, 0.005, 0.005, 0.1, 0.1, 0.1], dtype=np.float32),
        far_bound=np.array([0.02, 0.02, 0.02, 0.5, 0.5, 0.5], dtype=np.float32),
        near_distance=0.005,   # m: distance at/below which near_bound applies fully
        far_distance=0.30,    # m: distance at/above which far_bound applies fully
    ):
        assert far_distance > near_distance
        self.near_bound = near_bound
        self.far_bound = far_bound
        self.near_distance = near_distance
        self.far_distance = far_distance

    @staticmethod
    def _smoothstep(x):
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3 - 2 * x)  # zero derivative at both ends

    def get_bound(self, distance_to_goal: float) -> np.ndarray:
        t = (distance_to_goal - self.near_distance) / (self.far_distance - self.near_distance)
        t = self._smoothstep(t)
        return self.near_bound + t * (self.far_bound - self.near_bound)

    def scale_action(self, raw_action: np.ndarray, distance_to_goal: float) -> np.ndarray:
        """raw_action: SAC's tanh output in [-1, 1]^6. Returns actual Cartesian delta."""
        return raw_action * self.get_bound(distance_to_goal)