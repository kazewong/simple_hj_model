import mujoco

xml_string = """
<mujoco model="auto_jumping_humanoid_v2">

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.4 0.6 0.8" rgb2="0.1 0.1 0.2" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" width="512" height="512" mark="cross" markrgb="1 1 1"/>
    <material name="grid" texture="grid" texrepeat="2 2" texuniform="true" reflectance="0.2"/>
    <material name="body" rgba="0.8 0.6 0.4 1" specular="0.3" shininess="0.1"/>
    <material name="foot" rgba="0.2 0.2 0.2 1" specular="0.1"/>
    <material name="target" rgba="1 0.2 0.2 0.6"/>
    <material name="landing" rgba="0.2 1 0.2 0.4"/>
  </asset>
  

  <worldbody>

    <!-- Humanoid body -->
    <body name="torso" pos="0 0 1.2">
      <!-- Root joints for 2D planar motion -->
      <joint name="root_x" type="slide" axis="1 0 0" range="-10 10" damping="0.1"/>
      <joint name="root_z" type="slide" axis="0 0 1" range="-2 6" damping="0.2"/>
      <joint name="root_rot" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="0.5"/>
      
      <!-- Torso - just the main capsule -->
      <geom name="torso" type="capsule" fromto="0 0 -0.15 0 0 0.15" size="0.08" mass="15"/>
      
      <!-- Visual sites -->
      <site name="torso_center" pos="0 0 0" size="0.02" rgba="0 0 1 0.5"/>
      <site name="com_marker" pos="0 0 0" size="0.03" rgba="1 1 0 0.7"/>
      
      <!-- Thigh -->
      <body name="thigh" pos="0 0 -0.2">
        <joint name="hip" range="-1.2 2.0" stiffness="10" armature="0.01" damping="2.0"/>
        <geom name="thigh" type="capsule" fromto="0 0 0 0 0 -0.35" size="0.045" mass="4"/>
        <site name="hip_marker" pos="0 0 0" size="0.015" rgba="1 0 0 0.8"/>
        
        <!-- Shin -->
        <body name="shin" pos="0 0 -0.35">
          <joint name="knee" range="-0.2 2.2" stiffness="8" armature="0.01" damping="1.8"/>
          <geom name="shin" type="capsule" fromto="0 0 0 0 0 -0.35" size="0.035" mass="2"/>
          <site name="knee_marker" pos="0 0 0" size="0.015" rgba="0 1 0 0.8"/>
          
          <!-- Foot -->
          <body name="foot" pos="0 0 -0.35">
            <joint name="ankle" range="-1.0 1.0" stiffness="5" armature="0.01" damping="1.2"/>
            <!-- Foot geometry with better ground contact -->
            <geom name="foot_main" type="box" size="0.12 0.05 0.03" pos="0.05 0 -0.03" 
                  mass="0.8" material="foot" friction="1.8 0.1 0.05"/>
                
            <!-- Green foot sole rectangles -->
            <geom name="foot_sole_back" type="box" size="0.06 0.05 0.005" pos="0 0 -0.065" 
                  mass="0.01" rgba="0.2 0.8 0.2 1" friction="2.0 0.1 0.05"/>
            <geom name="foot_sole_front" type="box" size="0.06 0.05 0.005" pos="0.1 0 -0.065" 
                  mass="0.01" rgba="0.2 0.8 0.2 1" friction="2.0 0.1 0.05"/>
            
            <!-- Contact sites -->
            <site name="heel_contact" pos="-0.05 0 -0.055" size="0.01"/>
            <site name="toe_contact" pos="0.15 0 -0.05" size="0.01"/>
            <site name="foot_center" pos="0.05 0 -0.055" size="0.015" rgba="0 0 1 0.6"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  
  <!-- Position-controlled actuators for smooth jumping -->
  <actuator>
    <!-- Faster response - increase kp by 50-100% -->
    <position name="hip_ctrl" joint="hip" kp="300" kv="30" ctrlrange="-1.2 2.0"/>
    <position name="knee_ctrl" joint="knee" kp="240" kv="24" ctrlrange="-0.2 2.2"/>
    <position name="ankle_ctrl" joint="ankle" kp="160" kv="16" ctrlrange="-1.0 1.0"/>
  </actuator>
  
  <!-- Jump sequence keyframes -->
  <keyframe>
    <!-- Initial crouch position -->
    <key name="crouch" time="0.0" 
         qpos="0 1.2 0 -0.8 1.6 -0.4"
         ctrl="-0.8 1.6 -0.4"/>
    
    <!-- Pre-jump loading -->
    <key name="load" time="0.3" 
         qpos="0 1.2 0 -1.0 1.8 -0.5"
         ctrl="-1.0 1.8 -0.5"/>
    
    <!-- Jump extension -->
    <key name="extend" time="0.5" 
         qpos="0 1.2 0 0.3 0.2 -0.2"
         ctrl="0.3 0.2 -0.2"/>
    
    <!-- Flight position -->
    <key name="flight" time="0.8" 
         qpos="0 2.0 0 -0.2 0.8 0.1"
         ctrl="-0.2 0.8 0.1"/>
    
    <!-- Landing preparation -->
    <key name="land_prep" time="1.2" 
         qpos="0 1.5 0 -0.4 1.2 -0.2"
         ctrl="-0.4 1.2 -0.2"/>
  </keyframe>
  
  <!-- Comprehensive sensor suite -->
  <sensor>
    <!-- Joint positions and velocities -->
    <jointpos name="hip_pos" joint="hip"/>
    <jointpos name="knee_pos" joint="knee"/>
    <jointpos name="ankle_pos" joint="ankle"/>
    <jointpos name="root_x_pos" joint="root_x"/>
    <jointpos name="root_z_pos" joint="root_z"/>
    <jointpos name="root_rot_pos" joint="root_rot"/>
    
    <jointvel name="hip_vel" joint="hip"/>
    <jointvel name="knee_vel" joint="knee"/>
    <jointvel name="ankle_vel" joint="ankle"/>
    <jointvel name="root_x_vel" joint="root_x"/>
    <jointvel name="root_z_vel" joint="root_z"/>
    <jointvel name="root_rot_vel" joint="root_rot"/>
    
    <!-- Body state sensors -->
    <framepos name="torso_pos" objtype="body" objname="torso"/>
    <framequat name="torso_quat" objtype="body" objname="torso"/>
    <framelinvel name="torso_vel" objtype="body" objname="torso"/>
    <frameangvel name="torso_angvel" objtype="body" objname="torso"/>
    
    <!-- Contact sensors -->
    <touch name="heel_contact" site="heel_contact"/>
    <touch name="toe_contact" site="toe_contact"/>
    <touch name="foot_contact" site="foot_center"/>
    
    <!-- Center of mass -->
    <subtreecom name="body_com" body="torso"/>
    
    <!-- Custom sensors for jump control -->
    <user name="jump_phase" dim="1"/>
    <user name="jump_timer" dim="1"/>
    <user name="ground_contact" dim="1"/>
  </sensor>
</mujoco>
"""

class MultiSegmentModel:
    
    def __init__(self):
        self.spec = mujoco.MjSpec.from_string(xml_string)
