import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

rod_xml = """
  <worldbody>
    <!-- Falling cylindrical rod -->
    <body name="rod" pos="2 2 2">
      <freejoint name="rod_joint" />
      <geom
                name="rod_geom"
                type="cylinder"
                size="0.05 1.0"
                rgba="1 1 1 1"
                mass="2"
            />
      <inertial pos="0 0 0" mass="2" diaginertia="0.167 0.167 0.0025" />
    </body>

  </worldbody>

"""


class RodModel:
    position: np.ndarray = np.array([0., 0., 0.])
    length: float = 1.0  # Total length of the rod
    width: float = 0.05  # Radius of the rod
    mass: float = 2.0  # Mass of the rod
    spec: mujoco.MjSpec

    def __init__(
        self, length: float = 1.0, width: float = 0.05, mass: float = 2.0, euler_angle: np.ndarray = np.array([0., 0., 0.]), position: np.ndarray = np.array([0., 0., 0.])
    ):
        position[2] += length   # Adjust z position to account for rod length
        self.length = length
        self.width = width
        self.mass = mass
        self.euler_angle = euler_angle
        self.position = position

        spec = mujoco.MjSpec.from_string(rod_xml)
        spec.bodies[1].geoms[0].size[0] = self.width
        spec.bodies[1].geoms[0].size[1] = self.length
        spec.bodies[1].geoms[0].mass = self.mass
        spec.bodies[1].geoms[0].quat = Rotation.from_euler('xyz', self.euler_angle, degrees=True).as_quat()
        spec.bodies[1].pos[:] = self.position
        self.spec = spec
