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



import panel as pn
import numpy as np
import mujoco
import time
import re

pn.extension('tabulator')

class RodCollisionHeatmap:
    def __init__(self):
        # Parameter ranges
        self.param_ranges = {
            'hv': (3, 9),
            'vv': (-6, -1), 
            'd': (0, 2),
            'a': (-90, 0),
            'b': (-45, 0),
            'avm': (0, 6),
            'g': (-40, 0),
            'avn': (0, 6)
        }
        
        # Parameter labels for display
        self.param_labels = {
            'hv': 'Horizontal Velocity',
            'vv': 'Vertical Velocity',
            'd': 'Distance (Y-offset)',
            'a': 'Alpha (degrees)',
            'b': 'Beta (degrees)', 
            'avm': 'Angular Vel M-axis',
            'g': 'Gamma (degrees)',
            'avn': 'Angular Vel N-axis'
        }
        
        # Initialize widgets
        self.setup_widgets()
        
        # Initialize heatmap data
        self.heatmap_data = np.zeros((20, 20))

    def create_world_with_wall_height(self, wall_height):
        with open("world1.xml", 'r') as f:
            xml_content = f.read()
        
        wall_half_height = wall_height / 2
        wall_center_z = wall_half_height

        # Calculate side_detector height (wall_height - 0.1)
        wall_detector_height = wall_height - 0.1
        wall_detector_half_height = wall_detector_height / 2
        wall_detector_center_z = wall_detector_half_height

        # Update wall
        wall_pattern = r'<geom name="wall"[^>]*>'
        new_wall = f'<geom name="wall" type="box" size="5 0.03 {wall_half_height}" pos="0 0 {wall_center_z}" rgba="0.8 0.4 0.2 1"/>'
        modified_xml = re.sub(wall_pattern, new_wall, xml_content)

        # Update side_detector (0.1 shorter than wall)
        wall_detector_pattern = r'<geom name="side_detector"[^>]*>'
        new_wall_detector = f'<geom name="side_detector" type="box" size="5 0.1 {wall_detector_half_height}" pos="0 -0.15 {wall_detector_center_z}" rgba="0 1 0 0.3" contype="1" conaffinity="1"/>'
        modified_xml = re.sub(wall_detector_pattern, new_wall_detector, modified_xml)

        # Update far_side_detector (fixed height 0.2)
        far_detector_pattern = r'<geom name="far_side_detector"[^>]*>'
        new_far_detector = f'<geom name="far_side_detector" type="box" size="5 2 0.1" pos="0 -2 0.1" rgba="0 1 0 0.3" contype="1" conaffinity="1"/>'
        modified_xml = re.sub(far_detector_pattern, new_far_detector, modified_xml)

        model = mujoco.MjModel.from_xml_string(modified_xml)
        return model
    
    def setup_widgets(self):
        param_options = [(self.param_labels[k], k) for k in self.param_ranges.keys()]
        
        # Axis selection dropdowns
        self.x_axis_select = pn.widgets.Select(
            name="X-Axis Parameter", 
            value='g', 
            options=param_options,
            width=200
        )
        
        self.y_axis_select = pn.widgets.Select(
            name="Y-Axis Parameter", 
            value='avn', 
            options=param_options,
            width=200
        )
        
        # Create sliders for all parameters
        self.sliders = {}
        for param, (min_val, max_val) in self.param_ranges.items():
            default_val = (min_val + max_val) / 2
            
            self.sliders[param] = pn.widgets.FloatSlider(
                name=self.param_labels[param],
                start=min_val,
                end=max_val,
                value=default_val,
                step=(max_val - min_val) / 20,
                width=300
            )
        
        # Generate button
        self.generate_btn = pn.widgets.Button(
            name="Generate Heatmap", 
            button_type="primary",
            width=200
        )
        self.generate_btn.on_click(self.generate_heatmap)
        
        # Progress indicator
        self.progress = pn.indicators.Progress(
            name='Progress', 
            value=0, 
            max=1200,
            width=300
        )
        
        # Status text
        self.status_text = pn.pane.HTML("<b>Ready to generate heatmap</b>")
                
    def run_headless_simulation(self, hv, vv, d, a, b, avm, g, avn, wall_height):
        try:
            model = self.create_world_with_wall_height(wall_height)
            data = mujoco.MjData(model)
            
            # Initialize rod position and orientation
            rod_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "rod_geom")
            rod_half_length = model.geom_size[rod_geom_id][1]
            
            quaternion, omega, yz_angle, yz_projection_factor, angle_to_ground_rad = mn_rotation_to_quaternion(a, b, g, avm, avn)
            half_projection_length = rod_half_length * yz_projection_factor
            
            # Set initial position and velocity
            data.qpos[0:3] = [-1, d + half_projection_length * np.cos(np.deg2rad(90 - yz_angle)), rod_half_length*np.sin(angle_to_ground_rad) + 0.1]
            data.qpos[3:7] = quaternion
            data.qvel[0:3] = [hv*np.cos(np.deg2rad(a)), hv*np.sin(np.deg2rad(a)), vv]
            data.qvel[3:6] = omega
            
            # Simulation variables
            hit_wall = False
            hit_green_detector = False
            rod_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rod")
            
                    # Run simulation
            for _ in range(1000):  # Run for 1000 steps max
                mujoco.mj_step(model, data)
                rod_pos = data.qpos[0:3]
                
                # Check all contacts
                for i in range(data.ncon):
                    contact = data.contact[i]
                    geom1_id = contact.geom1
                    geom2_id = contact.geom2
                    body1_id = model.geom_bodyid[geom1_id]
                    body2_id = model.geom_bodyid[geom2_id]
                    
                    # Check if rod is involved in contact
                    if body1_id == rod_body_id or body2_id == rod_body_id:
                        geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
                        geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)
                        
                        # Check for detector contact
                        if (geom1_name and ("far_side_detector" in geom1_name or "side_detector" in geom1_name)) or \
                        (geom2_name and ("far_side_detector" in geom2_name or "side_detector" in geom2_name)):
                            hit_green_detector = True
                            return hit_green_detector
                        
                        # Check for wall contact
                        if (geom1_name and "wall" in geom1_name) or \
                           (geom2_name and "wall" in geom2_name):
                            hit_wall = True
                            return not hit_wall
            return False

        except:
            return False

    def generate_heatmap(self, event=None):
        x_param = self.x_axis_select.value[1]
        y_param = self.y_axis_select.value[1]
        
        if x_param == y_param:
            self.status_text.object = "<b style='color: red;'>Error: X and Y axes cannot be the same parameter!</b>"
            return
        
        start_time = time.time()
        self.status_text.object = "<b>Generating heatmap...</b>"
        self.progress.value = 0
        
        x_min, x_max = self.param_ranges[x_param]
        y_min, y_max = self.param_ranges[y_param]
        
        x_values = np.linspace(x_min, x_max, 20)
        y_values = np.linspace(y_min, y_max, 20)
        
        heatmap_data = np.zeros((20, 20))
        
        fixed_params = {}
        for param in self.param_ranges.keys():
            if param not in [x_param, y_param]:
                fixed_params[param] = self.sliders[param].value
        
        total_sims = 20 * 20 * 3
        sim_count = 0
        wall_heights = [1.7, 2.0, 2.3]
        
        for i, y_val in enumerate(y_values):
            for j, x_val in enumerate(x_values):
                params = fixed_params.copy()
                params[x_param] = x_val
                params[y_param] = y_val
                
                world_results = []
                for wall_height in wall_heights:
                    result = self.run_headless_simulation(
                        params['hv'], params['vv'], params['d'], params['a'],
                        params['b'], params['avm'], params['g'], params['avn'],
                        wall_height
                    )
                    world_results.append(result)
                    sim_count += 1
                    self.progress.value = sim_count
                
                world1_hit, world2_hit, world3_hit = world_results

                # Set heatmap colors based on results
                if world1_hit and world2_hit and world3_hit:
                    heatmap_data[i, j] = 0  # White - clears all walls
                elif world1_hit and world2_hit and not world3_hit:
                    heatmap_data[i, j] = 1  # Pink - clears up to 2.0m
                elif world1_hit and not world2_hit and not world3_hit:
                    heatmap_data[i, j] = 2  # Orange - clears only 1.5m
                else:
                    heatmap_data[i, j] = 3  # Red - clears no walls
                
                if sim_count % 60 == 0:
                    elapsed = time.time() - start_time
                    rate = sim_count / elapsed if elapsed > 0 else 0
                    eta = (total_sims - sim_count) / rate if rate > 0 else 0
                    self.status_text.object = f"<b>Progress: {sim_count}/{total_sims} ({sim_count/total_sims*100:.1f}%) - ETA: {eta:.1f}s</b>"
        
        self.heatmap_data = heatmap_data
        elapsed = time.time() - start_time
        self.status_text.object = f"<b style='color: green;'>Heatmap generated! ({sim_count} simulations in {elapsed:.1f}s)</b>"
        
        self.update_plot()
    
    def update_plot(self):
        x_param = self.x_axis_select.value[1]
        y_param = self.y_axis_select.value[1]
        
        x_min, x_max = self.param_ranges[x_param]
        y_min, y_max = self.param_ranges[y_param]
        
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['white', 'lightcoral', 'red', 'darkred']        
        custom_cmap = ListedColormap(colors)
        
        im = ax.imshow(self.heatmap_data, cmap=custom_cmap, aspect='auto', 
                    extent=[x_min, x_max, y_min, y_max], origin='lower',
                    vmin=0, vmax=3)
        
        ax.set_xlabel(f'{self.param_labels[x_param]} ({x_param})')
        ax.set_ylabel(f'{self.param_labels[y_param]} ({y_param})')
        ax.set_title('Rod Collision Heatmap')
        
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
        cbar.set_ticklabels(['Clears All', 'Clears 1.7m+2.0m', 'Clears 1.7m Only', 'Clears None'])
        
        plt.tight_layout()
        
        self.plot_pane.object = fig
        plt.close(fig)
    
    def create_dashboard(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, 'Click "Generate Heatmap" to start', 
                ha='center', va='center', transform=ax.transAxes, fontsize=16)
        ax.set_title('Rod Collision Heatmap')
        self.plot_pane = pn.pane.Matplotlib(fig, width=700, height=600)
        plt.close(fig)
        
        def get_slider_panel():
            x_param = self.x_axis_select.value[1] if isinstance(self.x_axis_select.value, tuple) else self.x_axis_select.value
            y_param = self.y_axis_select.value[1] if isinstance(self.y_axis_select.value, tuple) else self.y_axis_select.value
            
            slider_widgets = []
            for param, slider in self.sliders.items():
                if param not in [x_param, y_param]:
                    slider_widgets.append(slider)
            
            return pn.Column(*slider_widgets, width=350)
        
        def update_sliders(event=None):
            slider_panel[:] = [get_slider_panel()]
        
        self.x_axis_select.param.watch(update_sliders, 'value')
        self.y_axis_select.param.watch(update_sliders, 'value')
        
        slider_panel = pn.Column(get_slider_panel(), width=350)
        
        controls = pn.Column(
            "## Controls",
            self.x_axis_select,
            self.y_axis_select,
            "---",
            "## Fixed Parameters",
            slider_panel,
            "---",
            self.generate_btn,
            self.progress,
            self.status_text,
            width=350
        )
        
        dashboard = pn.Row(
            controls,
            pn.Spacer(width=20),
            self.plot_pane
        )
        
        return dashboard

# Create and serve the application
if __name__ == "__main__":
    app = RodCollisionHeatmap()
    dashboard = app.create_dashboard()
    pn.serve(dashboard, title="Rod Collision Heatmap", show=True, port=5009)
