import mujoco
import numpy as np
from coordinate_transform import mn_rotation_to_quaternion

pit_humanoid_xml = '''<mujoco model="Pit+Humanoid">
    <include file='C:/Users/eligi/revamp/src/hjsimulator/modular/pit.xml'/>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.4 0.6 0.8" rgb2="0.1 0.1 0.2" width="512" height="512"/>
    <material name="body_humanoid" rgba="0.8 0.6 0.4 1" specular="0.3" shininess="0.1"/>
    <material name="foot" rgba="0.2 0.2 0.2 1" specular="0.1"/>
    <material name="target_humanoid" rgba="1 0.2 0.2 0.6"/>
    <material name="landing_humanoid" rgba="0.2 1 0.2 0.4"/>
  </asset>
  
  <worldbody>
    <!-- Humanoid body with freejoint for full 3D motion -->
    <body name="torso" pos="0 0 1.2">
      <freejoint name="torso_free"/>
      
      <!-- Torso - main body capsule -->
      <geom name="torso" type="capsule" fromto="0 0 -0.3 0 0 0.3" size="0.16" 
            material="body_humanoid" mass="15"/>
      
      <!-- Visual sites for debugging -->
      <site name="torso_center" pos="0 0 0" size="0.04" rgba="0 0 1 0.5"/>
      <site name="com_marker" pos="0 0 0" size="0.06" rgba="1 1 0 0.7"/>
      
      <!-- Thigh -->
      <body name="thigh" pos="0 0 -0.4">
        <joint name="hip" type="hinge" axis="0 1 0" range="-1.2 2.0" 
               stiffness="10" armature="0.01" damping="2.0"/>
        <geom name="thigh" type="capsule" fromto="0 0 0 0 0 -0.7" size="0.09" 
              material="body_humanoid" mass="4"/>
        <site name="hip_marker" pos="0 0 0" size="0.03" rgba="1 0 0 0.8"/>
        
        <!-- Shin -->
        <body name="shin" pos="0 0 -0.7">
          <joint name="knee" type="hinge" axis="0 1 0" range="-0.2 2.2" 
                 stiffness="8" armature="0.01" damping="1.8"/>
          <geom name="shin" type="capsule" fromto="0 0 0 0 0 -0.7" size="0.07" 
                material="body_humanoid" mass="2"/>
          <site name="knee_marker" pos="0 0 0" size="0.03" rgba="0 1 0 0.8"/>
          
          <!-- Foot -->
          <body name="foot_humanoid" pos="0 0 -0.7">
            <joint name="ankle" type="hinge" axis="0 1 0" range="-1.0 1.0" 
                   stiffness="5" armature="0.01" damping="1.2"/>
            
            <!-- Main foot body -->
            <geom name="foot_main" type="box" size="0.24 0.1 0.06" pos="0.1 0 -0.06" 
                  material="foot" mass="0.8" friction="5 0.1 0.05"/>

          </body>
        </body>
      </body>
    </body>
  </worldbody>
  
  <!-- Position-controlled actuators -->
  <actuator>
    <position name="hip_ctrl" joint="hip" kp="300" kv="30" ctrlrange="-1.2 2.0"/>
    <position name="knee_ctrl" joint="knee" kp="240" kv="24" ctrlrange="-0.2 2.2"/>
    <position name="ankle_ctrl" joint="ankle" kp="160" kv="16" ctrlrange="-1.0 1.0"/>
  </actuator>
  
  
  <!-- Comprehensive sensor suite -->
  <sensor>
    <!-- Freejoint position and orientation -->
    <framepos name="torso_pos" objtype="body" objname="torso"/>
    <framequat name="torso_quat" objtype="body" objname="torso"/>
    <framelinvel name="torso_vel" objtype="body" objname="torso"/>
    <frameangvel name="torso_angvel" objtype="body" objname="torso"/>
    
    <!-- Joint positions and velocities -->
    <jointpos name="hip_pos" joint="hip"/>
    <jointpos name="knee_pos" joint="knee"/>
    <jointpos name="ankle_pos" joint="ankle"/>
    
    <jointvel name="hip_vel" joint="hip"/>
    <jointvel name="knee_vel" joint="knee"/>
    <jointvel name="ankle_vel" joint="ankle"/>
    
    <!-- Center of mass -->
    <subtreecom name="body_com" body="torso"/>
    
    <!-- Custom sensors for jump control -->
    <user name="jump_phase" dim="1"/>
    <user name="jump_timer" dim="1"/>
    <user name="ground_contact" dim="1"/>
  </sensor>
</mujoco>'''

import mujoco
import numpy as np

def quaternion_multiply(q1, q2):
    """Multiply quaternions q1 * q2, both in [w, x, y, z] format"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])

def rotate_quaternion_z(quaternion, angle_degrees):
    """
    Rotate a quaternion by angle_degrees around the Z-axis (clockwise when looking down)
    """
    angle_rad = np.deg2rad(-angle_degrees)  # Negative for clockwise
    
    # Create Z-axis rotation quaternion [w, x, y, z]
    z_rotation = np.array([
        np.cos(angle_rad/2),  # w
        0,                    # x
        0,                    # y
        np.sin(angle_rad/2)   # z
    ])
    
    # Apply Z rotation after the existing rotation
    return quaternion_multiply(quaternion, z_rotation)

def set_humanoid_initial_conditions(model, data, params):
    hv = params[0]
    vv = params[1]
    d = params[2]
    a = params[3]
    b = params[4]
    avm = params[5]
    g = params[6]
    avn = params[7]
    
    # Get the base quaternion from your existing rotation logic
    quaternion, omega, yz_angle, yz_projection_factor, angle_to_ground_rad = mn_rotation_to_quaternion(a, b, g, avm, avn)
    
    # Try removing this extra rotation first to see if that's the issue
    # Create rotation around the humanoid's own long axis
    # long_axis_rotation_angle = np.deg2rad(a)  # COMMENT THIS OUT
    # local_z_rotation = np.array([np.cos(long_axis_rotation_angle/2), 0, 0, np.sin(long_axis_rotation_angle/2)])
    
    # Apply the long-axis rotation AFTER the spatial orientation
    # final_quat = quaternion_multiply(quaternion, local_z_rotation)  # COMMENT THIS OUT
    final_quat = quaternion  # USE THIS INSTEAD
    
    # Rest of your code stays the same...
    data.qpos[0] = -1
    data.qpos[1] = d
    data.qpos[2] = 2.4
    data.qpos[3:7] = final_quat
    
    data.qpos[7] = np.deg2rad(-0.8)
    data.qpos[8] = np.deg2rad(1.6)
    data.qpos[9] = np.deg2rad(-0.4)
    
    data.qvel[0] = hv * np.cos(np.deg2rad(a))
    data.qvel[1] = hv * np.sin(np.deg2rad(a))
    data.qvel[2] = vv
    
    data.qvel[3:6] = omega
    
    data.qvel[6] = 0
    data.qvel[7] = 0
    data.qvel[8] = 0
    
    mujoco.mj_forward(model, data)