import mujoco
import numpy as np

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
      <geom name="torso" type="capsule" fromto="0 0 -0.15 0 0 0.15" size="0.08" 
            material="body_humanoid" mass="15"/>
      
      <!-- Visual sites for debugging -->
      <site name="torso_center" pos="0 0 0" size="0.02" rgba="0 0 1 0.5"/>
      <site name="com_marker" pos="0 0 0" size="0.03" rgba="1 1 0 0.7"/>
      
      <!-- Thigh -->
      <body name="thigh" pos="0 0 -0.2">
        <joint name="hip" type="hinge" axis="0 1 0" range="-1.2 2.0" 
               stiffness="10" armature="0.01" damping="2.0"/>
        <geom name="thigh" type="capsule" fromto="0 0 0 0 0 -0.35" size="0.045" 
              material="body_humanoid" mass="4"/>
        <site name="hip_marker" pos="0 0 0" size="0.015" rgba="1 0 0 0.8"/>
        
        <!-- Shin -->
        <body name="shin" pos="0 0 -0.35">
          <joint name="knee" type="hinge" axis="0 1 0" range="-0.2 2.2" 
                 stiffness="8" armature="0.01" damping="1.8"/>
          <geom name="shin" type="capsule" fromto="0 0 0 0 0 -0.35" size="0.035" 
                material="body_humanoid" mass="2"/>
          <site name="knee_marker" pos="0 0 0" size="0.015" rgba="0 1 0 0.8"/>
          
          <!-- Foot -->
          <body name="foot_humanoid" pos="0 0 -0.35">
            <joint name="ankle" type="hinge" axis="0 1 0" range="-1.0 1.0" 
                   stiffness="5" armature="0.01" damping="1.2"/>
            
            <!-- Main foot body -->
            <geom name="foot_main" type="box" size="0.12 0.05 0.03" pos="0.05 0 -0.03" 
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

def set_humanoid_initial_conditions(model, data, hv, vv, d, a, b, avm, g, avn):
    """
    Set initial conditions for humanoid - same orientation in space,
    but rotated around its own long axis to face movement direction
    """
    
    # Get the base quaternion from your existing rotation logic
    quaternion, omega, yz_angle, yz_projection_factor, angle_to_ground_rad = mn_rotation_to_quaternion(a, b, g, avm, avn)
    
    # Create rotation around the humanoid's own long axis
    long_axis_rotation_angle = np.deg2rad(a)  # or whatever angle you want
    local_z_rotation = np.array([np.cos(long_axis_rotation_angle/2), 0, 0, np.sin(long_axis_rotation_angle/2)])
    
    # Apply the long-axis rotation AFTER the spatial orientation
    final_quat = quaternion_multiply(quaternion, local_z_rotation)
    
    # Set torso position (freejoint position - first 3 qpos elements)
    data.qpos[0] = -1  # x position
    data.qpos[1] = d   # y position (distance parameter)
    data.qpos[2] = 1.2 # z position (humanoid standing height)
    
    # Set torso orientation (freejoint quaternion - next 4 qpos elements)
    data.qpos[3:7] = final_quat  # [w, x, y, z]
    
    # Set joint angles (elements 7, 8, 9 - default crouch position)
    data.qpos[7] = np.deg2rad(-0.8)   # hip angle
    data.qpos[8] = np.deg2rad(1.6)    # knee angle  
    data.qpos[9] = np.deg2rad(-0.4)   # ankle angle
    
    # Set torso linear velocities (freejoint linear velocity - first 3 qvel elements)
    data.qvel[0] = hv * np.cos(np.deg2rad(a))  # x velocity component
    data.qvel[1] = hv * np.sin(np.deg2rad(a))  # y velocity component
    data.qvel[2] = vv                          # z velocity (vertical)
    
    # Set torso angular velocities (freejoint angular velocity - next 3 qvel elements)
    data.qvel[3:6] = omega  # [omega_x, omega_y, omega_z]
    
    # Set joint velocities (elements 6, 7, 8 - start with zero)
    data.qvel[6] = 0  # hip angular velocity
    data.qvel[7] = 0  # knee angular velocity
    data.qvel[8] = 0  # ankle angular velocity
    
    # Update simulation state
    mujoco.mj_forward(model, data)

def quaternion_multiply(q1, q2):
    """Multiply quaternions q1 * q2, both in [w, x, y, z] format"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])

def mn_rotation_to_quaternion(alpha_deg, beta_deg, gamma_deg, omega_m, omega_n):

    alpha = np.radians(alpha_deg)
    beta = np.radians(beta_deg)
    gamma = np.radians(gamma_deg)
    
    n_axis = np.array([np.cos(alpha), np.sin(alpha), 0])
    
    m_axis = np.array([-np.sin(alpha), np.cos(alpha), 0])
    
    def axis_angle_to_quat(axis, angle):
        axis = axis / np.linalg.norm(axis)
        half_angle = angle / 2
        w = np.cos(half_angle)
        xyz = axis * np.sin(half_angle)
        return np.array([w, xyz[0], xyz[1], xyz[2]])
    
    def axis_angle_to_rotation_matrix(axis, angle):
        axis = axis / np.linalg.norm(axis)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        x, y, z = axis
        return np.array([
            [cos_a + x*x*(1-cos_a), x*y*(1-cos_a) - z*sin_a, x*z*(1-cos_a) + y*sin_a],
            [y*x*(1-cos_a) + z*sin_a, cos_a + y*y*(1-cos_a), y*z*(1-cos_a) - x*sin_a],
            [z*x*(1-cos_a) - y*sin_a, z*y*(1-cos_a) + x*sin_a, cos_a + z*z*(1-cos_a)]
        ])
    
    R_m = axis_angle_to_rotation_matrix(m_axis, beta)
    R_n = axis_angle_to_rotation_matrix(n_axis, gamma)
    
    R_combined = R_n @ R_m
    
    local_rod_axis = np.array([0, 0, 1])
    
    world_rod_axis = R_combined @ local_rod_axis
    
    yz_angle_rad = np.arctan2(world_rod_axis[1], world_rod_axis[2])
    yz_angle_deg = np.degrees(yz_angle_rad)

    yz_components = np.array([world_rod_axis[1], world_rod_axis[2]])
    yz_projection_factor = np.linalg.norm(yz_components)

    angle_to_ground_rad = np.arcsin(abs(world_rod_axis[2]))
    
    def rotation_matrix_to_quat(R):
        """Convert rotation matrix to quaternion"""
        trace = np.trace(R)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (R[2,1] - R[1,2]) / s
            y = (R[0,2] - R[2,0]) / s
            z = (R[1,0] - R[0,1]) / s
        else:
            if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
                s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
                w = (R[2,1] - R[1,2]) / s
                x = 0.25 * s
                y = (R[0,1] + R[1,0]) / s
                z = (R[0,2] + R[2,0]) / s
            elif R[1,1] > R[2,2]:
                s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
                w = (R[0,2] - R[2,0]) / s
                x = (R[0,1] + R[1,0]) / s
                y = 0.25 * s
                z = (R[1,2] + R[2,1]) / s
            else:
                s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
                w = (R[1,0] - R[0,1]) / s
                x = (R[0,2] + R[2,0]) / s
                y = (R[1,2] + R[2,1]) / s
                z = 0.25 * s
        return np.array([w, x, y, z])
    
    final_quat = rotation_matrix_to_quat(R_combined)
    
    omega_world = omega_m * m_axis + omega_n * n_axis
    
    return final_quat, omega_world, yz_angle_deg, yz_projection_factor, angle_to_ground_rad