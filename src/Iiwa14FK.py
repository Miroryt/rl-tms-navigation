from scipy.spatial.transform import Rotation as R
import numpy as np
from urchin import URDF

"""A method to change the path to meshes when reading joint info from iiwa14.urdf"""
def fix_package_uris(urdf_path, package_map):
    with open(urdf_path, 'r') as f:
        xml_data = f.read()

    for pkg, path in package_map.items():
        xml_data = xml_data.replace(f'package://{pkg}', path)

    fixed_path = urdf_path.replace('.urdf', '_fixed.urdf')
    with open(fixed_path, 'w') as f:
        f.write(xml_data)

    return fixed_path

def build_chain(robot, base, ee):
    """
    Return ordered list of joints and links from base -> ee.
    """
    joint_chain = []
    link_chain = []

    # Start at the end effector
    current_link = ee
    link_chain.append(current_link)

    while current_link != base:
        # Find the joint that connects this link to its parent
        parent_joint = None
        for j in robot.joints:
            if j.child == current_link:
                parent_joint = j
                break
        if parent_joint is None:
            raise ValueError(f"No parent joint found for link {current_link}")
        joint_chain.insert(0, parent_joint.name)
        current_link = parent_joint.parent
        link_chain.insert(0, current_link)

    return link_chain, joint_chain

"""Class to compute the Forward Kinematic of iiwa 14"""
class Iiwa14FK:
    def __init__(self, urdf_path, base_link='lbr_link_0', ee_link='lbr_link_ee'):
        self.robot = URDF.load(urdf_path)
        self.base = base_link
        self.ee = ee_link
        # Precompute the ordered joint list from base->ee
        self.chain_links, self.chain_joints = build_chain(self.robot, self.base, self.ee)
        # Store joint axes and fixed transforms in base frame order
        self.joint_axes = []
        self.X_T = []   # fixed transforms from parent->child when q=0
        for jn in self.chain_joints:
            j = self.robot.joint_map[jn]
            # parent->child transform with q=0
            T_fixed = j.origin
            self.X_T.append(T_fixed)
            if j.joint_type != 'revolute':
                self.joint_axes.append(None)
            else:
                # axis in joint frame coordinates
                self.joint_axes.append(np.array(j.axis, dtype=float))

    def fk(self, q):
        """
        ---------
        Calculate FK from joints
        This method returns EE position from lbr_link_ee (Gazebo UI shows EE_position to be lbr_link_7 (3.5 cm less in z-coordinate than lbr_link_ee)
        q: (7,) radians for the 7 chain joints (in chain order)
        ---------
        """
        T = np.eye(4)
        qi = 0
        for jn, T_fixed, axis in zip(self.chain_joints, self.X_T, self.joint_axes):
            T = T @ T_fixed
            if axis is not None:           # revolute
                ax = axis / np.linalg.norm(axis)
                th = q[qi]
                c, s = np.cos(th), np.sin(th)
                """Rodrigues' rotation formula"""
                K = np.array([[0, -ax[2], ax[1]],
                              [ax[2], 0, -ax[0]],
                              [-ax[1], ax[0], 0]])
                Rj = np.eye(3) + s*K + (1-c)*(K@K)
                Tj = np.eye(4); Tj[:3,:3] = Rj
                T = T @ Tj
                qi += 1
        pos = T[:3, 3]
        quat = R.from_matrix(T[:3,:3]).as_quat()  # x,y,z,w
        return pos, quat

    def get_link_masses(self):
        """Extract masses in chain order (link_0 -> link_ee)"""
        link_masses = []
        for ln in self.chain_links:
            if ln in self.robot.link_map:
                link = self.robot.link_map[ln]
                if link.inertial is not None:
                    link_masses.append(link.inertial.mass)
                else:
                    link_masses.append(0.0) #No mass defined

        return link_masses