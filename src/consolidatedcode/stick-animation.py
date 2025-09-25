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

def check_wall_contact(model, data):
    
    rod_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rod")
    
    # Check all contacts in the simulation
    for i in range(data.ncon):
        contact = data.contact[i]
        
        # Get the geom IDs involved in this contact
        geom1_id = contact.geom1
        geom2_id = contact.geom2
        
        # Get the body IDs for these geometries
        body1_id = model.geom_bodyid[geom1_id]
        body2_id = model.geom_bodyid[geom2_id]
        
        # Check if one of the bodies is the rod
        if body1_id == rod_body_id or body2_id == rod_body_id:
            # Get the names of the geometries to identify what the rod is touching
            geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
            geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)
            
            # Check if either geometry is a wall (assuming walls have "wall" in their name)
            if (geom1_name and "wall" in geom1_name.lower()) or \
               (geom2_name and "wall" in geom2_name.lower()):
                return 0  # Contact with wall detected
    return

def run_rod_simulation(hv, vv, d, a, b, avm, g, avn):
    
    # Load the world model
    model = mujoco.MjModel.from_xml_path("world1.xml")
    data = mujoco.MjData(model)

    rod_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "rod_geom")
    rod_half_length = model.geom_size[rod_geom_id][1]

    quaternion, omega, yz_angle, yz_projection_factor, angle_to_ground_rad = mn_rotation_to_quaternion(a, b, g, avm, avn)
    half_projection_length = rod_half_length * yz_projection_factor

    data.qpos[0:3] = [-1, d + half_projection_length * np.cos(np.deg2rad(90 - yz_angle)), rod_half_length*np.sin(angle_to_ground_rad) + 0.1]
    data.qpos[3:7] = quaternion
    data.qvel[0:3] = [hv*np.cos(np.deg2rad(a)), -hv*np.sin(np.deg2rad(a)), vv]
    data.qvel[3:6] = omega

    # Create viewer with keyboard callback
    paused = True

    def key_callback(keycode):
        nonlocal paused  # Use nonlocal instead of global
        if keycode == 32:  # Spacebar key code
            paused = not paused
            if paused:
                print("Simulation PAUSED - Press SPACEBAR to resume")
            else:
                print("Simulation STARTED - Press SPACEBAR to pause")

    # Launch the viewer (non-passive for keyboard support)
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        # Set camera to look at the origin
        viewer.cam.lookat[0] = 0    # Look at X=0 (origin)
        viewer.cam.lookat[1] = 0    # Look at Y=0 (origin)  
        viewer.cam.lookat[2] = 1.5  # Look at Z=1.5 (slightly above ground)
        
        viewer.cam.distance = 12     # Distance from the look-at point
        viewer.cam.elevation = -20   # Look down at -20 degrees
        
        print("Rod is ready! Press SPACEBAR in the viewer window to start/pause simulation")
        
        # Initialize time for camera rotation
        start_time = time.time()
        
        while viewer.is_running():
            if not paused:
                mujoco.mj_step(model, data)
                wall_contact_status = check_wall_contact(model, data)
                if wall_contact_status == 0:
                    print("Rod has contacted a wall! Simulation paused.")
                    paused = True
            
            # Update camera azimuth to circle around origin
            current_time = time.time()
            elapsed_time = current_time - start_time
            rotation_speed = 10  # degrees per second
            
            viewer.sync()
            time.sleep(0.01)  # Prevent excessive CPU usage when paused

# Usage:
run_rod_simulation(
    hv=7,
    vv=-5.5,
    d=0.7,
    a=45,
    b=10,
    avm=-5,
    g=-30,
    avn=7
)
