import numpy as np

# black box methods below, uncleaned code
def mn_rotation_to_quaternion(alpha_deg, beta_deg, gamma_deg, omega_m, omega_n):

    alpha = np.radians(alpha_deg)
    beta = np.radians(beta_deg)
    gamma = np.radians(gamma_deg)
    
    n_axis = np.array([np.cos(alpha), np.sin(alpha), 0])
    
    m_axis = np.array([-np.sin(alpha), np.cos(alpha), 0])
    
    
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