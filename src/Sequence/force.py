import os
import time
from dataclasses import dataclass
import numpy as np
import mujoco
import mujoco.viewer


# ----------------------------
# Configuration / dataclasses
# ----------------------------

@dataclass
class InitialConditions:
    """Starting state"""
    # Root pose
    x_pos: float = -1.0
    z_pos: float = 0.2
    rotation: float = -0.5
    # Joint angles
    hip_angle: float = -0.8
    knee_angle: float = 1.0
    ankle_angle: float = -0.4
    # Velocities
    x_vel: float = 2.0
    z_vel: float = 0.0
    rot_vel: float = 1.5
    hip_vel: float = 0.0
    knee_vel: float = 0.0
    ankle_vel: float = 0.0


class JumpController:
    """Simple phase-based controller; returns desired controls (one per actuator)."""
    def __init__(self):
        self.enabled = True
        self.jump_completed = False
        self.jump_start_time = 0.0

    def reset_jump(self):
        self.enabled = True
        self.jump_completed = False
        self.jump_start_time = 0.0
        if hasattr(self, 'jump_state'):
            delattr(self, 'jump_state')
        if hasattr(self, 'phase_start_time'):
            delattr(self, 'phase_start_time')
        print("Jump sequence reset - ready to jump!")

    def get_targets(self, t: float, ground_contact: bool) -> tuple[float, float, float]:
        """Return hip/knee/ankle command targets."""
        if not self.enabled or self.jump_completed:
            return 0.0, 0.0, 0.0

        # lazy-init
        if not hasattr(self, 'jump_state'):
            self.jump_state = 'crouch'
            self.phase_start_time = t

        phase_time = t - self.phase_start_time

        if self.jump_state == 'crouch':
            if phase_time > 0.3:
                self.jump_state = 'load'
                self.phase_start_time = t
            return -0.6, 1.4, -0.3

        elif self.jump_state == 'load':
            if phase_time > 0.2:
                self.jump_state = 'jump'
                self.phase_start_time = t
            return -0.8, 1.6, -0.2

        elif self.jump_state == 'jump':
            if not ground_contact:
                self.jump_state = 'flight'
                self.phase_start_time = t
            return 0.2, 0.1, 0.4

        elif self.jump_state == 'flight':
            if ground_contact:
                self.jump_state = 'landing'
                self.phase_start_time = t
            return -0.3, 1.0, 0.1

        elif self.jump_state == 'landing':
            if phase_time > 0.5:
                self.jump_completed = True
            return -0.4, 1.0, -0.1

        return 0.0, 0.0, 0.0


# ----------------------------
# Simulation wrapper
# ----------------------------

class HumanoidSimulation:
    """MuJoCo sim + viewer + logging of joint forces."""

    def __init__(self, model_path: str):
        # Load model/data
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Components
        self.initial_conditions = InitialConditions()
        self.jump_controller = JumpController()

        # runtime flags
        self.paused = False
        self.reset_requested = False
        self.step_once = False

        # Indices and setup
        self._resolve_indices()
        self.reset_simulation()

    # ---------- indexing / setup ----------

    def _resolve_indices(self):
        """Resolve joint/DoF/actuator indices once, by name."""
        m = self.model

        # JOINT IDs
        self.joint_id = {
            'root_x': mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'root_x'),
            'root_z': mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'root_z'),
            'root_rot': mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'root_rot'),
            'hip': mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'hip'),
            'knee': mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'knee'),
            'ankle': mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'ankle'),
        }

        # Address in qpos / qvel (per joint)
        self.qpos_adr = {k: m.jnt_qposadr[jid] for k, jid in self.joint_id.items()}
        self.qvel_adr = {k: m.jnt_dofadr[jid] for k, jid in self.joint_id.items()}

        # DoF indices for convenience (same as qvel_adr for hinge/slide)
        self.dof_idx = {k: m.jnt_dofadr[jid] for k, jid in self.joint_id.items()}

        # ACTUATORS
        self.actuator_idx = {
            'hip': mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, 'hip_ctrl'),
            'knee': mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, 'knee_ctrl'),
            'ankle': mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, 'ankle_ctrl'),
        }

        # Cache ctrl ranges for clamping
        self.ctrl_range = {}
        for name, aidx in self.actuator_idx.items():
            if aidx >= 0 and m.actuator_ctrllimited[aidx]:
                lo, hi = m.actuator_ctrlrange[aidx]
            else:
                lo, hi = -np.inf, np.inf
            self.ctrl_range[name] = (float(lo), float(hi))

        # Try to find optional contact sensors (by common names). Fallback to ncon>0.
        self._contact_sensor_slices = []
        for sname in ['foot_contact_L', 'foot_contact_R', 'contact_L', 'contact_R',
                      'foot_L_force', 'foot_R_force', 'foot_force']:
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, sname)
            if sid != -1:
                adr = m.sensor_adr[sid]
                dim = m.sensor_dim[sid]
                self._contact_sensor_slices.append((adr, dim))
        if self._contact_sensor_slices:
            print(f"[INFO] Using {len(self._contact_sensor_slices)} sensor slice(s) for ground contact.")
        else:
            print("[INFO] No named contact sensors found. Falling back to `data.ncon > 0`.")

    # ---------- state management ----------

    def reset_simulation(self):
        """Apply ICs, reset controller, clear logs."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)

        # Apply initial positions (use qpos addresses!)
        d.qpos[self.qpos_adr['root_x']] = self.initial_conditions.x_pos
        d.qpos[self.qpos_adr['root_z']] = self.initial_conditions.z_pos
        d.qpos[self.qpos_adr['root_rot']] = self.initial_conditions.rotation
        d.qpos[self.qpos_adr['hip']] = self.initial_conditions.hip_angle
        d.qpos[self.qpos_adr['knee']] = self.initial_conditions.knee_angle
        d.qpos[self.qpos_adr['ankle']] = self.initial_conditions.ankle_angle

        # Apply initial velocities (use qvel addresses!)
        d.qvel[self.qvel_adr['root_x']] = self.initial_conditions.x_vel
        d.qvel[self.qvel_adr['root_z']] = self.initial_conditions.z_vel
        d.qvel[self.qvel_adr['root_rot']] = self.initial_conditions.rot_vel
        d.qvel[self.qvel_adr['hip']] = self.initial_conditions.hip_vel
        d.qvel[self.qvel_adr['knee']] = self.initial_conditions.knee_vel
        d.qvel[self.qvel_adr['ankle']] = self.initial_conditions.ankle_vel

        self.jump_controller.reset_jump()
        mujoco.mj_forward(m, d)

        # initialize logging buffers
        self.force_log = []  # rows: [t, hip_act, hip_con, knee_act, knee_con, ankle_act, ankle_con, hip_net, knee_net, ankle_net]
        self.max_abs = {'hip': 0.0, 'knee': 0.0, 'ankle': 0.0}
        self._log_saved = False

        print("Simulation reset - new jump sequence starting!")
        self.print_current_state()

    def print_current_state(self):
        d = self.data
        print("\nCurrent State:")
        print(f"Position: x={d.qpos[self.qpos_adr['root_x']]:.3f}, "
              f"z={d.qpos[self.qpos_adr['root_z']]:.3f}, "
              f"rot={d.qpos[self.qpos_adr['root_rot']]:.3f}")
        print(f"Joints: hip={d.qpos[self.qpos_adr['hip']]:.3f}, "
              f"knee={d.qpos[self.qpos_adr['knee']]:.3f}, "
              f"ankle={d.qpos[self.qpos_adr['ankle']]:.3f}")
        print(f"Velocities: x_vel={d.qvel[self.qvel_adr['root_x']]:.3f}, "
              f"z_vel={d.qvel[self.qvel_adr['root_z']]:.3f}")
        print(f"Time: {d.time:.3f}s")

    # ---------- contact + logging ----------

    def _has_ground_contact(self) -> bool:
        """Return True if feet (or anything) are in contact with ground."""
        d = self.data
        if self._contact_sensor_slices:
            for adr, dim in self._contact_sensor_slices:
                if np.max(d.sensordata[adr:adr + dim]) > 0.1:
                    return True
            return False
        # Fallback: any contact at all
        return d.ncon > 0

    def _log_forces_per_step(self):
        """Log generalized joint forces and update running maxima."""
        d = self.data
        di = self.dof_idx  # dof indices

        # Per-DoF generalized forces
        hip_act   = float(d.qfrc_actuator[di['hip']])
        hip_con   = float(d.qfrc_constraint[di['hip']])
        knee_act  = float(d.qfrc_actuator[di['knee']])
        knee_con  = float(d.qfrc_constraint[di['knee']])
        ankle_act = float(d.qfrc_actuator[di['ankle']])
        ankle_con = float(d.qfrc_constraint[di['ankle']])

        hip_net   = hip_act + hip_con
        knee_net  = knee_act + knee_con
        ankle_net = ankle_act + ankle_con

        self.force_log.append([
            d.time,
            hip_act, hip_con,
            knee_act, knee_con,
            ankle_act, ankle_con,
            hip_net, knee_net, ankle_net
        ])

        # Running maxima of absolute net torque/force at each DoF
        self.max_abs['hip']   = max(self.max_abs['hip'],   abs(hip_net))
        self.max_abs['knee']  = max(self.max_abs['knee'],  abs(knee_net))
        self.max_abs['ankle'] = max(self.max_abs['ankle'], abs(ankle_net))

        # Auto-save once jump completes
        if self.jump_controller.jump_completed and not self._log_saved:
            self._save_force_log()
            self._log_saved = True
            print(f"[LOG] Max |net generalized force| during jump: "
                  f"hip={self.max_abs['hip']:.2f}, "
                  f"knee={self.max_abs['knee']:.2f}, "
                  f"ankle={self.max_abs['ankle']:.2f}")

    def _save_force_log(self, folder="logs"):
        os.makedirs(folder, exist_ok=True)
        # Timestamped filename
        import datetime, csv
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(folder, f"jump_forces_{ts}.csv")
        header = [
            "time",
            "hip_qfrc_act", "hip_qfrc_con",
            "knee_qfrc_act", "knee_qfrc_con",
            "ankle_qfrc_act", "ankle_qfrc_con",
            "hip_net", "knee_net", "ankle_net"
        ]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(self.force_log)
        print(f"[LOG] Saved force log → {path}")

    # ---------- control & keys ----------

    def _clamp_ctrl(self, name: str, val: float) -> float:
        lo, hi = self.ctrl_range[name]
        return float(np.clip(val, lo, hi))

    def control_callback(self, model, data):
        """Compute controls, write to actuators, and (optionally) do other per-step tasks."""
        ground_contact = self._has_ground_contact()

        # Get desired control signals from your controller
        hip_u, knee_u, ankle_u = self.jump_controller.get_targets(data.time, ground_contact)

        # Write controls with clamping
        data.ctrl[self.actuator_idx['hip']]   = self._clamp_ctrl('hip',   hip_u)
        data.ctrl[self.actuator_idx['knee']]  = self._clamp_ctrl('knee',  knee_u)
        data.ctrl[self.actuator_idx['ankle']] = self._clamp_ctrl('ankle', ankle_u)

    def key_callback(self, keycode):
        """Keyboard controls for viewer."""
        if keycode == 32:  # Space
            self.paused = not self.paused
            print(f"Simulation {'paused' if self.paused else 'resumed'}")

        elif keycode == ord('R'):
            self.reset_requested = True

        elif keycode == ord('T'):   # step once (when paused)
            if self.paused:
                self.step_once = True
                print("Stepping once...")

        elif keycode == ord('J'):
            self.jump_controller.enabled = not self.jump_controller.enabled
            print(f"Jump controller {'enabled' if self.jump_controller.enabled else 'disabled'}")
            if self.jump_controller.enabled and self.jump_controller.jump_completed:
                print("Jump already completed - press 'R' to jump again")

        elif keycode == ord('N'):   # new jump without full reset
            if self.jump_controller.jump_completed:
                self.jump_controller.reset_jump()
                # Also reset the logging for a fresh jump if desired
                self.force_log.clear()
                self.max_abs = {'hip': 0.0, 'knee': 0.0, 'ankle': 0.0}
                self._log_saved = False
                print("New jump sequence started!")
            else:
                print("Current jump still in progress...")

        elif keycode == ord('P'):   # print state + current max loads
            self.print_current_state()
            print(f"Max |net generalized force| so far: "
                  f"hip={self.max_abs['hip']:.2f}, "
                  f"knee={self.max_abs['knee']:.2f}, "
                  f"ankle={self.max_abs['ankle']:.2f}")

        # Initial condition tweaks (apply with 'R')
        elif keycode == ord('Q'):
            self.initial_conditions.z_pos += 0.2
            print(f"Initial height: {self.initial_conditions.z_pos:.2f} (press 'R' to apply)")

        elif keycode == ord('A'):
            self.initial_conditions.z_pos = max(0.5, self.initial_conditions.z_pos - 0.2)
            print(f"Initial height: {self.initial_conditions.z_pos:.2f} (press 'R' to apply)")

        elif keycode == ord('W'):
            self.initial_conditions.x_pos += 0.5
            print(f"Initial X position: {self.initial_conditions.x_pos:.2f} (press 'R' to apply)")

        elif keycode == ord('S'):
            self.initial_conditions.x_pos -= 0.5
            print(f"Initial X position: {self.initial_conditions.x_pos:.2f} (press 'R' to apply)")

        elif keycode == ord('U'):
            self.initial_conditions.x_vel += 0.5
            print(f"Initial X velocity: {self.initial_conditions.x_vel:.2f} (press 'R' to apply)")

        elif keycode == ord('I'):
            self.initial_conditions.x_vel -= 0.5
            print(f"Initial X velocity: {self.initial_conditions.x_vel:.2f} (press 'R' to apply)")

        elif keycode == ord('O'):
            self.initial_conditions.z_vel += 0.5
            print(f"Initial Z velocity: {self.initial_conditions.z_vel:.2f} (press 'R' to apply)")

        elif keycode == ord('L'):
            self.initial_conditions.z_vel -= 0.5
            print(f"Initial Z velocity: {self.initial_conditions.z_vel:.2f} (press 'R' to apply)")

        elif keycode == ord('Z'):
            self.initial_conditions.x_vel = 0.0
            self.initial_conditions.z_vel = 0.0
            self.initial_conditions.rot_vel = 0.0
            print("All initial velocities reset to zero (press 'R' to apply)")

        elif keycode == ord('X'):
            self.initial_conditions.x_pos = 0.0
            self.initial_conditions.z_pos = 1.2
            print("Position reset to origin (0, 1.2) (press 'R' to apply)")

        elif keycode == ord('H'):
            self.print_help()

    def print_help(self):
        print("\n" + "="*60)
        print("KEYS:")
        print("SPACE : Pause/Resume")
        print("R     : Reset simulation and start new jump")
        print("N     : Start new jump (without full reset)")
        print("T     : Step once (when paused)")
        print("J     : Toggle jump controller")
        print("P     : Print state + max joint forces so far")
        print("H     : Show this help")
        print("\nINITIAL CONDITIONS (press 'R' to apply):")
        print("Q/A   : Increase/Decrease initial height")
        print("W/S   : Move initial X forward/back")
        print("U/I   : Add forward/backward initial velocity")
        print("O/L   : Add upward/downward initial velocity")
        print("="*60)

    # ---------- main loop ----------

    def run(self):
        print("Starting MuJoCo Humanoid Simulation...")
        self.print_help()

        with mujoco.viewer.launch_passive(self.model, self.data,
                                          key_callback=self.key_callback) as viewer:
            # Camera
            viewer.cam.distance = 4.0
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 90
            viewer.cam.lookat = np.array([0.0, 0.0, 1.0])

            while viewer.is_running():
                step_start = time.time()

                if self.reset_requested:
                    self.reset_simulation()
                    self.reset_requested = False

                # step physics
                if not self.paused or self.step_once:
                    self.control_callback(self.model, self.data)
                    mujoco.mj_step(self.model, self.data)

                    # log AFTER stepping
                    self._log_forces_per_step()

                    self.step_once = False

                viewer.sync()

                # real-time pacing
                dt = self.model.opt.timestep - (time.time() - step_start)
                if dt > 0:
                    time.sleep(dt)


# ----------------------------
# Entry point
# ----------------------------

def main():
    model_path = r"C:/Users/eligi/Downloads/simple_hj_model/src/Sequence/bio-model.xml"

    if not os.path.exists(model_path):
        print(f"Error: XML file not found at: {model_path}")
        return

    try:
        print(f"Loading model from: {model_path}")
        sim = HumanoidSimulation(model_path)
        sim.run()
    except Exception as e:
        print(f"Error loading or running simulation: {e}")


if __name__ == "__main__":
    main()
