import mujoco

xml_string = """
<mujoco model="rod">
  <worldbody>
    <!-- Falling cylindrical rod -->
    <body name="rod" pos="2 2 2">
      <freejoint name="rod_joint" />
      <geom
                name="rod_geom"
                type="cylinder"
                size="0.05 0.96"
                rgba="1 1 1 1"
                mass="2"
            />
      <inertial pos="0 0 0" mass="2" diaginertia="0.167 0.167 0.0025" />
    </body>

  </worldbody>

</mujoco>
"""


class RodModel:
    length: float = 0.96  # Total length of the rod
    width: float = 0.05  # Radius of the rod
    mass: float = 2.0  # Mass of the rod
    spec: mujoco.MjSpec

    def __init__(
        self, length: float = 0.96, width: float = 0.05, mass: float = 2.0
    ):
        self.length = length
        self.width = width
        self.mass = mass
        
        spec = mujoco.MjSpec.from_string(xml_string)
        spec.bodies[1].geoms[0].size[0] = self.width
        spec.bodies[1].geoms[0].size[1] = self.length
        spec.bodies[1].geoms[0].mass = self.mass
        self.spec = spec
