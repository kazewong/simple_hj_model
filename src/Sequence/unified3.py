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
    
    # Convert rotation matrix to quaternion
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

def quat_multiply(q1, q2):
    """
    Multiply two quaternions: q1 * q2
    Remember: q2 is applied first, then q1
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def check_wall_contact(model, data):
    try:
        leg_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "leg_body")
    except:
        print("Warning: 'leg_body' not found in model")
        return 1
    
    # Check all contacts in the simulation
    for i in range(data.ncon):
        contact = data.contact[i]
        
        # Get the geom IDs involved in this contact
        geom1_id = contact.geom1
        geom2_id = contact.geom2
        
        # Get the body IDs for these geometries
        body1_id = model.geom_bodyid[geom1_id]
        body2_id = model.geom_bodyid[geom2_id]
        
        # Check if one of the bodies is the leg (or any of its child bodies)
        leg_bodies = []
        try:
            leg_bodies.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "leg_body"))
            leg_bodies.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "thigh"))
            leg_bodies.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "shin"))
            leg_bodies.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot"))
        except:
            pass  # Some bodies might not exist
        
        if body1_id in leg_bodies or body2_id in leg_bodies:
            # Get the names of the geometries to identify what the leg is touching
            geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
            geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)
            
            # Check if either geometry is a wall (assuming walls have "wall" in their name)
            if (geom1_name and "wall" in geom1_name.lower()) or \
               (geom2_name and "wall" in geom2_name.lower()):
                return 0  # Contact with wall detected
    return 1  # No contact

def run_leg_simulation(hv, vv, d, a, b, avm, g, avn):
    """
    FINAL VERSION: Both m-n parameters AND foot direction work correctly
    
    Parameters:
    - hv: horizontal velocity
    - vv: vertical velocity  
    - d: lateral displacement
    - a: alpha angle - defines m-n axes AND foot pointing direction
    - b: beta rotation (around m-axis perpendicular to direction a)
    - avm: angular velocity around m-axis
    - g: gamma rotation (around n-axis along direction a) 
    - avn: angular velocity around n-axis
    """
    
    try:
        model = mujoco.MjModel.from_xml_path(r"C:\Users\eligi\Downloads\simple_hj_model\src\Sequence\unified.xml")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    data = mujoco.MjData(model)
    leg_half_length = 1.0

    print(f"\n=== Leg Simulation Parameters ===")
    print(f"Velocities: hv={hv}, vv={vv}")
    print(f"Position: d={d}")
    print(f"Rotations: α={a}° (m-n axes & foot direction), β={b}°, γ={g}°")
    print(f"Angular velocities: ωm={avm}, ωn={avn}")

    # Step 1: Get m-n rotations using the ORIGINAL system (this works correctly)
    mn_quaternion, omega, yz_angle, yz_projection_factor, angle_to_ground_rad = mn_rotation_to_quaternion(a, b, g, avm, avn)
    
    # Step 2: Create foot alignment quaternion
    # The leg's foot points in +X direction by default
    # We want it to point in direction 'a'
    # We need to rotate by 'a' around the Z-axis, but in the CORRECT direction
    
    # Test both directions to see which one works:
    foot_alignment_angle = -np.radians(a)  # Try positive first
    foot_alignment_quat = np.array([
        np.cos(foot_alignment_angle / 2),  # w
        0,                                 # x
        0,                                 # y  
        np.sin(foot_alignment_angle / 2)   # z
    ])
    
    # Step 3: Combine the rotations
    # We want: m-n rotations happen in the coordinate system where foot points in direction 'a'
    # So: foot alignment FIRST, then m-n rotations
    final_quaternion = quat_multiply(mn_quaternion, foot_alignment_quat)
    
    print(f"M-N axes: m⟂{a}°, n∥{a}°")
    print(f"Foot pointing: {a}°")
    
    # Use EXACTLY the same calculations as original working code
    half_projection_length = leg_half_length * yz_projection_factor
    initial_pos = np.array([
        -1, 
        d + half_projection_length * np.cos(np.radians(90 - yz_angle)), 
        leg_half_length * np.sin(angle_to_ground_rad) + 0.1
    ])
    
    vel_x = hv * np.cos(np.radians(a))
    vel_y = -hv * np.sin(np.radians(a))
    
    print(f"World velocity: ({vel_x:.2f}, {vel_y:.2f}, {vv})")
    print(f"Initial position: ({initial_pos[0]:.2f}, {initial_pos[1]:.2f}, {initial_pos[2]:.2f})")

    # Set initial state
    data.qpos[0:3] = initial_pos
    data.qpos[3:7] = final_quaternion
    data.qvel[0:3] = [vel_x, vel_y, vv]
    data.qvel[3:6] = omega

    # Set leg joints to neutral
    try:
        data.qpos[7] = 0.0   # hip neutral
        data.qpos[8] = 0.0   # knee neutral  
        data.qpos[9] = 0.0   # ankle neutral
        print("Leg joints set to neutral positions")
    except:
        print("Warning: Could not find leg joints")

    # Viewer setup
    paused = True

    def key_callback(keycode):
        nonlocal paused
        if keycode == 32:  # Spacebar
            paused = not paused
            print("Simulation PAUSED" if paused else "Simulation STARTED")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.lookat[0] = 0
        viewer.cam.lookat[1] = 0  
        viewer.cam.lookat[2] = 1.5
        viewer.cam.distance = 12
        viewer.cam.elevation = -20
        
        print("\n=== Controls ===")
        print("Press SPACEBAR to start/pause simulation")
        
        start_time = time.time()
        
        while viewer.is_running():
            if not paused:
                mujoco.mj_step(model, data)
                wall_contact_status = check_wall_contact(model, data)
                if wall_contact_status == 0:
                    elapsed_time = time.time() - start_time
                    print(f"\n*** WALL CONTACT at t={elapsed_time:.2f}s ***")
                    paused = True
            
            viewer.sync()
            time.sleep(0.01)

# Test to verify BOTH issues are fixed:
if __name__ == "__main__":
    print("Testing BOTH m-n parameters AND foot direction...")
    
    # This should:
    # 1. Have foot pointing at 45°
    # 2. Have beta rotation around m-axis (perpendicular to 45°)  
    # 3. Have gamma rotation around n-axis (along 45°)
    run_leg_simulation(hv=7, vv=-5.5, d=0.7, a=45, b=0, avm=0, g=-20, avn=7)