import mujoco
import mujoco.viewer
import numpy as np
import time
import os
from dataclasses import dataclass

@dataclass
class InitialConditions:
    """Store initial conditions for the humanoid"""
    # Position
    x_pos: float = -1
    z_pos: float = 0.2
    rotation: float = -0.4
    
    # Joint angles
    hip_angle: float = -0.8
    knee_angle: float = 1.0
    ankle_angle: float = -0.4
    
    # Velocities
    x_vel: float = 2
    z_vel: float = 0.0
    rot_vel: float = 1
    hip_vel: float = 0
    knee_vel: float = 0
    ankle_vel: float = 0

class JumpController:
    """Single-jump controller that waits for reset"""
    
    def __init__(self):
        self.enabled = True
        self.jump_completed = False
        
    def reset_jump(self):
        """Reset jump state - called when 'R' is pressed"""
        self.jump_completed = False
        self.jump_start_time = 0.0
        
        # Reset all state machine variables
        if hasattr(self, 'jump_state'):
            delattr(self, 'jump_state')
        if hasattr(self, 'phase_start_time'):
            delattr(self, 'phase_start_time')
        
        print("Jump sequence reset - ready to jump!")
        
    def get_targets(self, time: float, ground_contact: bool) -> tuple:
        """Sensor-based jump - reacts to ground contact"""
        if not self.enabled:
            return 0.0, 0.0, 0.0
        
        if self.jump_completed:
            return 0.0, 0.0, 0.0
        
        # STATE MACHINE based on contact and time
        if not hasattr(self, 'jump_state'):
            self.jump_state = 'crouch'
            self.phase_start_time = time
        
        phase_time = time - self.phase_start_time
        
        if self.jump_state == 'crouch':
            if phase_time > 0.3:  # Crouch for 0.3s
                self.jump_state = 'load'
                self.phase_start_time = time
            return -0.6, 1.4, -0.3
        
        elif self.jump_state == 'load':
            if phase_time > 0.1:  # Load for 0.2s then jump
                self.jump_state = 'jump'
                self.phase_start_time = time
            return -0.8, 1.6, -0.2
        
        elif self.jump_state == 'jump':
            if not ground_contact:  # After load, before leaving ground
                self.jump_state = 'flight'
                self.phase_start_time = time
            return 0.2, 0.1, 0.4
        
        elif self.jump_state == 'flight':
            if ground_contact:  # Landing
                self.jump_state = 'landing'
                self.phase_start_time = time
            return -0.3, 1.0, 0.1
        
        elif self.jump_state == 'landing':
            if phase_time > 0.5:  # Stable for 0.5s
                self.jump_completed = True
            return -0.4, 1.0, -0.1
        
        return 0.0, 0.0, 0.0


class HumanoidSimulation:
    """Main simulation class with interactive viewer"""
    
    def __init__(self, model_path: str):
        # Load model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Initialize components
        self.initial_conditions = InitialConditions()
        self.jump_controller = JumpController()
        
        # Simulation state
        self.paused = False
        self.reset_requested = False
        self.step_once = False
        
        # Get joint and actuator indices
        self._get_indices()
        
        # Set initial state
        self.reset_simulation()
        
    def _get_indices(self):
        """Get indices for joints and actuators"""
        self.joint_indices = {
            'root_x': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'root_x'),
            'root_z': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'root_z'),
            'root_rot': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'root_rot'),
            'hip': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'hip'),
            'knee': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'knee'),
            'ankle': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'ankle'),
        }
        
        self.actuator_indices = {
            'hip': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'hip_ctrl'),
            'knee': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'knee_ctrl'),
            'ankle': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'ankle_ctrl'),
        }
    
    def reset_simulation(self):
        """Reset simulation to initial conditions"""
        mujoco.mj_resetData(self.model, self.data)
        
        # Set positions
        self.data.qpos[self.joint_indices['root_x']] = self.initial_conditions.x_pos
        self.data.qpos[self.joint_indices['root_z']] = self.initial_conditions.z_pos
        self.data.qpos[self.joint_indices['root_rot']] = self.initial_conditions.rotation
        self.data.qpos[self.joint_indices['hip']] = self.initial_conditions.hip_angle
        self.data.qpos[self.joint_indices['knee']] = self.initial_conditions.knee_angle
        self.data.qpos[self.joint_indices['ankle']] = self.initial_conditions.ankle_angle
        
        # Set velocities
        self.data.qvel[self.joint_indices['root_x']] = self.initial_conditions.x_vel
        self.data.qvel[self.joint_indices['root_z']] = self.initial_conditions.z_vel
        self.data.qvel[self.joint_indices['root_rot']] = self.initial_conditions.rot_vel
        self.data.qvel[self.joint_indices['hip']] = self.initial_conditions.hip_vel
        self.data.qvel[self.joint_indices['knee']] = self.initial_conditions.knee_vel
        self.data.qvel[self.joint_indices['ankle']] = self.initial_conditions.ankle_vel
        
        # Reset jump controller
        self.jump_controller.reset_jump()
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
        print("Simulation reset - new jump sequence starting!")
        self.print_current_state()
    
    def print_current_state(self):
        """Print current state for debugging"""
        print(f"\nCurrent State:")
        print(f"Position: x={self.data.qpos[0]:.2f}, z={self.data.qpos[1]:.2f}, rot={self.data.qpos[2]:.2f}")
        print(f"Joints: hip={self.data.qpos[3]:.2f}, knee={self.data.qpos[4]:.2f}, ankle={self.data.qpos[5]:.2f}")
        print(f"Velocities: x_vel={self.data.qvel[0]:.2f}, z_vel={self.data.qvel[1]:.2f}")
        print(f"Time: {self.data.time:.2f}s")
    
    def control_callback(self, model, data):
        """Control callback for automatic jumping"""
        # Get ground contact
        ground_contact = False
        if hasattr(data, 'sensordata') and len(data.sensordata) > 15:
            ground_contact = max(data.sensordata[15:18]) > 0.1
        
        # Get control targets
        hip_target, knee_target, ankle_target = self.jump_controller.get_targets(
            data.time, ground_contact
        )
        
        # Set control signals
        data.ctrl[self.actuator_indices['hip']] = hip_target
        data.ctrl[self.actuator_indices['knee']] = knee_target
        data.ctrl[self.actuator_indices['ankle']] = ankle_target
    
    def key_callback(self, keycode):
        """Handle keyboard input - using non-conflicting keys"""
        if keycode == 32:  # Spacebar - pause/unpause
            self.paused = not self.paused
            print(f"Simulation {'paused' if self.paused else 'resumed'}")
            
        elif keycode == ord('R'):  # R - reset
            self.reset_requested = True
            
        elif keycode == ord('3'):  # step once
            if self.paused:
                self.step_once = True
                print("Stepping once...")
                
        elif keycode == ord('Z'):  # J - toggle jump controller
            self.jump_controller.enabled = not self.jump_controller.enabled
            status = "enabled" if self.jump_controller.enabled else "disabled"
            print(f"Jump controller {status}")
            if self.jump_controller.enabled and self.jump_controller.jump_completed:
                print("Jump already completed - press 'R' to jump again")
                
        elif keycode == ord('P'):  # P - print state
            self.print_current_state()
            if self.jump_controller.jump_completed:
                print("Status: Jump completed - press 'R' for new jump")
            else:
                print("Status: Jump in progress...")
            
        elif keycode == ord('H'):  # Help
            self.print_help()
    
    def print_help(self):
        """Print help information with new working keys"""
        print("\n" + "="*60)
        print("WORKING KEYBOARD CONTROLS (MuJoCo-Compatible):")
        print("="*60)
        print("SPACE - Pause/Resume simulation")
        print("R     - Reset simulation and start new jump")
        print("3     - Step once (when paused)")
        print("Z     - Toggle automatic jump controller")
        print("P     - Print current state and jump status")
        print("H     - Show this help")
        print()
        print("JUMP BEHAVIOR:")
        print("- Robot jumps once, then holds landing position")
        print("- Press 'R' to reset and jump again")
        print("="*60)
    
    def run(self):
        """Run the simulation with viewer"""
        print("Starting MuJoCo Humanoid Simulation...")
        self.print_help()
        
        with mujoco.viewer.launch_passive(self.model, self.data, 
                                        key_callback=self.key_callback) as viewer:
            # Set camera for better view
            viewer.cam.distance = 4.0
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 90
            viewer.cam.lookat = np.array([0.0, 0.0, 1.0])
            
            while viewer.is_running():
                step_start = time.time()
                
                # Handle reset request
                if self.reset_requested:
                    self.reset_simulation()
                    self.reset_requested = False
                
                # Simulation step
                if not self.paused or self.step_once:
                    # Apply control
                    self.control_callback(self.model, self.data)
                    
                    # Step simulation
                    mujoco.mj_step(self.model, self.data)
                    self.step_once = False
                
                # Update viewer
                viewer.sync()
                
                # Maintain real-time execution
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

def main():
    """Main function"""
    model_path = r"C:/Users/eligi/Downloads/simple_hj_model/src/Mujoco Simulations/3body.xml"
    
    # Check if file exists
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