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
    
    def __init__(self):
        self.spec = mujoco.MjSpec.from_string(xml_string)
