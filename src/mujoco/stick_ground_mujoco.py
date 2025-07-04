

import itertools
import mujoco
from mujoco import mjx
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

class JumpSimulator:
    def __init__(self, horizontal_velocity, vertical_velocity, incident_angle, angular_velocity):
        self.hv = horizontal_velocity
        self.vv = vertical_velocity
        self.ia = incident_angle
        self.av = angular_velocity

        self.num_steps = 600
        self.positions = []
        self.num_contacts = []
        self.saved_qpos = []
        self.max_height_after = 0
        self.max_height_after_idx = None
        self.first_contact = None
        self.first_contact_end = None

        self._setup_sim()

    def _setup_sim(self):
        xml = """
        <mujoco>
            <option gravity="0 0 -9.81"/>
            <worldbody>
                <camera name="sideview" pos="2 0 0.5" euler="0 90 90"/>
                <geom name="ground" type="plane" size="2 2 0.1" pos="0 0 0" rgba="0 0.6 0 1" friction="1 0.005 0.0001"/>
                <body name="rodd" pos="0 -0.5 1">
                    <freejoint/>
                    <geom name="rod" type="cylinder" pos="0 0 0" size="0.001 0.25" rgba="1 1 1 1" solref="0.0001 0.001" solimp="0.99 0.99 0.01" friction="1 0.005 0.0001"/>
                </body>
            </worldbody>
        </mujoco>
        """
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        # self.renderer = mujoco.Renderer(self.model)
        self.model_mjx = mjx.put_model(self.model)
        self.data_mjx = mjx.put_data(self.model, self.data)

        self.data.qpos[0:3] = [0, -0.5, 0.01 + np.cos(self.ia)/4]
        self.data.qpos[3:7] = [np.cos(self.ia / 2), np.sin(self.ia / 2), 0, 0]
        self.data.qvel[0:6] = [0, self.hv, self.vv, self.av, 0, 0]

        self.rod_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "rod")

    def run(self):
        for _ in range(self.num_steps):
            mujoco.mj_step(self.model, self.data)
            self.positions.append(self.data.xipos[1].copy())
            self.num_contacts.append(self.data.ncon)
            self.saved_qpos.append(self.data.qpos.copy())

        self.positions = np.array(self.positions)
        self.saved_qpos = np.array(self.saved_qpos)
        self._detect_contacts()
        self._compute_max_height()

    def _detect_contacts(self):
        for i, n in enumerate(self.num_contacts):
            if n > 0 and self.first_contact is None:
                self.first_contact = i
            if self.first_contact is not None and n == 0:
                self.first_contact_end = i
                break

    def _compute_max_height(self):
        if self.first_contact_end is not None:
            z_after = self.positions[self.first_contact_end:, 2]
            if len(z_after) > 0:
                self.max_height_after = np.max(z_after)
                self.max_height_after_idx = np.argmax(z_after) + self.first_contact_end

    def get_max_height(self):
        return self.max_height_after

    def print_summary(self):
        if self.first_contact is not None and self.first_contact_end is not None:
            duration = self.first_contact_end - self.first_contact
            print(f"First collision lasts {duration} steps")
        elif self.first_contact is not None:
            print(f"First collision starts at step {self.first_contact} and lasts until the end of the simulation")
        else:
            print("No collision detected in the simulation")

        if self.max_height_after_idx is not None:
            print(f"Max height after collision: {self.max_height_after:.4f} m at step {self.max_height_after_idx}")
        else:
            print("No valid height found after collision.")

    def analyze_orientation_at_events(self):
        if self.first_contact is None or self.max_height_after_idx is None:
            return

        for step, label in [(self.first_contact, "contact"), (self.max_height_after_idx, "peak")]:
            quat = self.saved_qpos[step, 3:7]
            quat_xyzw = [quat[1], quat[2], quat[3], quat[0]]
            rot_matrix = R.from_quat(quat_xyzw).as_matrix()
            axis_yz = rot_matrix[1:, 2]
            angle_rad = np.arctan2(axis_yz[1], axis_yz[0])
            angle_deg = np.degrees(angle_rad)
            print(f"Angle at {label} (step {step}): {angle_deg:.2f} degrees")

    def summarize_parameters(self):
        print("Simulation Parameters:")
        print(f"  Horizontal Velocity: {self.hv:.2f} m/s")
        print(f"  Vertical Velocity: {self.vv:.2f} m/s")
        print(f"  Incident Angle: {np.degrees(self.ia):.2f} degrees")
        print(f"  Angular Velocity: {self.av:.2f} rad/s")

def iterate(hv_min, hv_max, vv_min, vv_max, ia_min, ia_max, av_min, av_max):
    horizontal_velocity_range = np.arange(hv_min, hv_max, 0.1)
    vertical_velocity_range = np.arange(vv_min, vv_max, 0.1)
    incident_angle_range = np.deg2rad(np.arange(ia_min, ia_max, 1))
    angular_velocity_range = np.arange(av_min, av_max, 1)

    top_jump = 0.0
    jumps = 0
    for hv, vv, ia, av in itertools.product(
            horizontal_velocity_range,
            vertical_velocity_range,
            incident_angle_range,
            angular_velocity_range):
        sim = JumpSimulator(hv, vv, ia, av)
        sim.run()
        max_height_after = sim.get_max_height()
        jumps+=1
        if max_height_after > top_jump:
            top_jump = max_height_after
            top_jump_hv = hv
            top_jump_vv = vv
            top_jump_ia = ia
            top_jump_av = av

    best_sim = JumpSimulator(top_jump_hv, top_jump_vv, top_jump_ia, top_jump_av)
    best_sim.run()
    print(f"Total jumps run: {jumps}")
    print(f"Top jump height: {top_jump:.4f} m")
    best_sim.summarize_parameters()
    best_sim.print_summary()
    best_sim.analyze_orientation_at_events()
    best_sim.animate_jump()





