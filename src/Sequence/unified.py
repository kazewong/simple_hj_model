import numpy as np

def mn_rotation_to_quaternion(alpha_deg, beta_deg, gamma_deg, omega_m=0, omega_n=0):
    """
    Convert rotations and angular velocities in your m-n coordinate system
    
    alpha_deg: angle of n-axis relative to x-axis
    beta_deg: rotation about m-axis
    gamma_deg: rotation about n-axis
    omega_m: angular velocity about m-axis (rad/s) - optional
    omega_n: angular velocity about n-axis (rad/s) - optional
    
    Returns: (quaternion, angular_velocity_world, yz_plane_angle)
    """
    
    # Convert to radians
    alpha = np.radians(alpha_deg)
    beta = np.radians(beta_deg)
    gamma = np.radians(gamma_deg)
    
    # Define your m and n axes in terms of x,y,z coordinates
    # n-axis direction vector
    n_axis = np.array([np.cos(alpha), np.sin(alpha), 0])
    
    # m-axis direction vector (perpendicular to n, in xy plane)
    m_axis = np.array([-np.sin(alpha), np.cos(alpha), 0])
    
    # Create rotation quaternions for each axis
    def axis_angle_to_quat(axis, angle):
        """Convert axis-angle to quaternion"""
        axis = axis / np.linalg.norm(axis)  # normalize
        half_angle = angle / 2
        w = np.cos(half_angle)
        xyz = axis * np.sin(half_angle)
        return np.array([w, xyz[0], xyz[1], xyz[2]])
    
    # Create rotation matrices for each axis (for direct calculation)
    def axis_angle_to_rotation_matrix(axis, angle):
        """Convert axis-angle to rotation matrix"""
        axis = axis / np.linalg.norm(axis)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        x, y, z = axis
        return np.array([
            [cos_a + x*x*(1-cos_a), x*y*(1-cos_a) - z*sin_a, x*z*(1-cos_a) + y*sin_a],
            [y*x*(1-cos_a) + z*sin_a, cos_a + y*y*(1-cos_a), y*z*(1-cos_a) - x*sin_a],
            [z*x*(1-cos_a) - y*sin_a, z*y*(1-cos_a) + x*sin_a, cos_a + z*z*(1-cos_a)]
        ])
    
    # Rotation matrices for each rotation
    R_m = axis_angle_to_rotation_matrix(m_axis, beta)
    R_n = axis_angle_to_rotation_matrix(n_axis, gamma)
    
    # Combined rotation matrix (first m, then n)
    R_combined = R_n @ R_m
    
    # Rod's local axis (along z in local coordinates)
    local_rod_axis = np.array([0, 0, 1])
    
    # Transform to world coordinates
    world_rod_axis = R_combined @ local_rod_axis
    
    # Calculate angle in yz plane
    yz_angle_rad = np.arctan2(world_rod_axis[1], world_rod_axis[2])
    yz_angle_deg = np.degrees(yz_angle_rad)

    yz_components = np.array([world_rod_axis[1], world_rod_axis[2]])
    yz_projection_factor = np.linalg.norm(yz_components)  # sqrt(y² + z²)

    angle_to_ground_rad = np.arcsin(abs(world_rod_axis[2]))  # |z_component|
    
    # Still need quaternion for MuJoCo, so convert from rotation matrix
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
    
    # Calculate angular velocity
    omega_world = omega_m * m_axis + omega_n * n_axis
    
    return final_quat, omega_world, yz_angle_deg, yz_projection_factor, angle_to_ground_rad


import mujoco
import mujoco.viewer
import time
import numpy as np

# Keep ALL the stick code exactly the same, just change these lines:

def check_wall_contact(model, data):
    # Change this line:
    leg_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "leg_body")  # was "rod"
    
    # Rest exactly the same
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_id = contact.geom1
        geom2_id = contact.geom2
        body1_id = model.geom_bodyid[geom1_id]
        body2_id = model.geom_bodyid[geom2_id]
        
        if body1_id == leg_body_id or body2_id == leg_body_id:
            geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
            geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)
            
            if (geom1_name and "wall" in geom1_name.lower()) or \
               (geom2_name and "wall" in geom2_name.lower()):
                return 0
    return

def run_leg_simulation(hv, vv, d, a, b, avm, g, avn):
    model = mujoco.MjModel.from_xml_path(r"C:\Users\eligi\Downloads\simple_hj_model\src\Sequence\unified.xml")
    data = mujoco.MjData(model)

    leg_half_length = 1.0
    
    # Calculate velocity direction
    vel_x = hv * np.cos(np.deg2rad(a))
    vel_y = -hv * np.sin(np.deg2rad(a))
    
    # Calculate angle to rotate foot toward velocity direction
    velocity_angle = np.arctan2(vel_y, vel_x)  # Angle of velocity vector
    
    # Create quaternion to rotate around Z-axis to point foot toward velocity
    quat_w = np.cos(velocity_angle / 2)
    quat_z = np.sin(velocity_angle / 2)
    velocity_quat = np.array([quat_w, 0, 0, quat_z])
    
    # Get the original m-n rotation
    quaternion, omega, yz_angle, yz_projection_factor, angle_to_ground_rad = mn_rotation_to_quaternion(a, b, g, avm, avn)
    
    # Combine the rotations: first the m-n rotation, then the velocity alignment
    def quat_multiply(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    final_quat = quat_multiply(velocity_quat, quaternion)
    
    # Position and velocity
    half_projection_length = leg_half_length * yz_projection_factor
    data.qpos[0:3] = [-1, d + half_projection_length * np.cos(np.deg2rad(90 - yz_angle)), leg_half_length*np.sin(angle_to_ground_rad) + 0.1]
    data.qpos[3:7] = final_quat  # Use combined rotation
    data.qvel[0:3] = [vel_x, vel_y, vv]
    data.qvel[3:6] = omega

    # Rest of the code stays the same...

    # Keep ALL the viewer code exactly the same
    paused = True

    def key_callback(keycode):
        nonlocal paused
        if keycode == 32:
            paused = not paused
            if paused:
                print("Simulation PAUSED - Press SPACEBAR to resume")
            else:
                print("Simulation STARTED - Press SPACEBAR to pause")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.lookat[0] = 0
        viewer.cam.lookat[1] = 0  
        viewer.cam.lookat[2] = 1.5
        viewer.cam.distance = 9
        viewer.cam.elevation = -15
        
        print("Leg is ready! Press SPACEBAR to start/pause")
        
        start_time = time.time()
        
        while viewer.is_running():
            if not paused:
                mujoco.mj_step(model, data)
                wall_contact_status = check_wall_contact(model, data)
                if wall_contact_status == 0:
                    print("Leg has contacted a wall! Simulation paused.")
                    paused = True
            
            viewer.sync()
            time.sleep(0.01)

# Same usage
run_leg_simulation(hv=7, vv=-4, d=0.5, a=45, b=0, avm=0, g=-20, avn=0)