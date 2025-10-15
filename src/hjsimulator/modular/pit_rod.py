import mujoco
import numpy as np
from coordinate_transform import mn_rotation_to_quaternion

rod_half_length = 1

pit_rod_xml = f'''<mujoco model="PitRod">
    <include file='C:/Users/eligi/revamp/src/hjsimulator/modular/pit.xml'/>
  <worldbody>
    <!-- Falling cylindrical rod -->
    <body name="rod" pos="2 2 10">
      <freejoint name="rod_joint" />
      <geom
                name="rod_geom"
                type="cylinder"
                size="0.05 {rod_half_length}"
                rgba="1 1 1 1"
                mass="2"
            />
      <inertial pos="0 0 0" mass="2" diaginertia="0.167 0.167 0.0025" />
    </body>

  </worldbody>
</mujoco>'''

def set_rod_initial_conditions(model, data, params):

    hv = params[0]
    vv = params[1]
    d = params[2]
    a = params[3]
    b = params[4]
    avm = params[5]
    g = params[6]
    avn = params[7]
    
    quaternion, omega, yz_angle, yz_projection_factor, angle_to_ground_rad = mn_rotation_to_quaternion(a, b, g, avm, avn)
    half_projection_length = rod_half_length * yz_projection_factor

    # initial conditions:
    data.qpos[0:3] = [-1, d + half_projection_length * np.cos(np.deg2rad(90 - yz_angle)), rod_half_length*np.sin(angle_to_ground_rad) + 0.1]
    data.qpos[3:7] = quaternion
    data.qvel[0:3] = [hv*np.cos(np.deg2rad(a)), hv*np.sin(np.deg2rad(a)), vv]
    data.qvel[3:6] = omega
    
    mujoco.mj_forward(model, data)