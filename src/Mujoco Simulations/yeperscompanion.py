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
    x_pos: float = 0.0
    z_pos: float = 1.2
    rotation: float = 0.0
    
    # Joint angles
    hip_angle: float = -0.8
    knee_angle: float = 1.6
    ankle_angle: float = -0.4
    
    # Velocities
    x_vel: float = 0.0
    z_vel: float = 0.0
    rot_vel: float = 0.0
    hip_vel: float = 0.0
    knee_vel: float = 0.0
    ankle_vel: float = 0.0

class JumpController:
    """Single-jump controller that waits for reset"""
    
    def __init__(self):
        self.jump_cycle_time = 3.0
        self.jump_height_multiplier = 1.0
        self.jump_forward_velocity = 0.5
        self.enabled = True
        self.jump_completed = False
        self.jump_start_time = 0.0
        
    def reset_jump(self):
        """Reset jump state - called when 'R' is pressed"""
        self.jump_completed = False
        self.jump_start_time = 0.0
        print("Jump sequence reset - ready to jump!")
        
    def get_targets(self, time: float, ground_contact: bool) -> tuple:
        """Get joint targets - only jump once until reset"""
        if not self.enabled:
            return 0.0, 0.0, 0.0
        
        # If jump is completed, hold landing position
        if self.jump_completed:
            return -0.4, 1.2, -0.2
        
        # Set jump start time on first call
        if self.jump_start_time == 0.0:
            self.jump_start_time = time
            
        # Calculate time since jump started
        jump_time = time - self.jump_start_time
        
        # Single jump sequence
        if jump_time < 0.5:  # Crouch phase
            hip = -0.8 * self.jump_height_multiplier
            knee = 1.6 * self.jump_height_multiplier
            ankle = -0.4
        elif jump_time < 0.8:  # Load phase  
            hip = -1.0 * self.jump_height_multiplier
            knee = 1.8 * self.jump_height_multiplier
            ankle = -0.5
        elif jump_time < 1.0:  # Jump phase
            hip = 0.3 * self.jump_height_multiplier
            knee = 0.2
            ankle = -0.2
        # elif jump_time < 1.8:  # Flight phase
        #     hip = -0.2
        #     knee = 0.8
        #     ankle = 0.1
        # elif jump_time < 2.5:  # Landing phase
        #     hip = -0.2
        #     knee = 0.8
        #     ankle = 0.1
        else:  # Jump completed
            self.jump_completed = True
            hip = -0.2
            knee = 0.8
            ankle = 0.1
            print("Jump sequence completed! Press 'R' to jump again.")
            
        return hip, knee, ankle

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
        """Handle keyboard input"""
        if keycode == 32:  # Spacebar - pause/unpause
            self.paused = not self.paused
            print(f"Simulation {'paused' if self.paused else 'resumed'}")
            
        elif keycode == ord('R'):  # R - reset
            self.reset_requested = True
            
        elif keycode == ord('S'):  # S - step once
            if self.paused:
                self.step_once = True
                
        elif keycode == ord('J'):  # J - toggle jump controller
            self.jump_controller.enabled = not self.jump_controller.enabled
            status = "enabled" if self.jump_controller.enabled else "disabled"
            print(f"Jump controller {status}")
            if self.jump_controller.enabled and self.jump_controller.jump_completed:
                print("Jump already completed - press 'R' to jump again")
                
        elif keycode == ord('N'):  # N - start new jump without full reset
            if self.jump_controller.jump_completed:
                self.jump_controller.reset_jump()
                print("New jump sequence started!")
            else:
                print("Current jump still in progress...")
                
        elif keycode == ord('P'):  # P - print state
            self.print_current_state()
            if self.jump_controller.jump_completed:
                print("Status: Jump completed - press 'R' for new jump")
            else:
                print("Status: Jump in progress...")
            
        # Adjust initial conditions
        elif keycode == ord('1'):  # Increase initial height
            self.initial_conditions.z_pos += 0.2
            print(f"Initial height: {self.initial_conditions.z_pos:.2f}")
            
        elif keycode == ord('2'):  # Decrease initial height
            self.initial_conditions.z_pos = max(0.5, self.initial_conditions.z_pos - 0.2)
            print(f"Initial height: {self.initial_conditions.z_pos:.2f}")
            
        elif keycode == ord('3'):  # Move initial position forward
            self.initial_conditions.x_pos += 0.5
            print(f"Initial X position: {self.initial_conditions.x_pos:.2f}")
            
        elif keycode == ord('4'):  # Move initial position backward
            self.initial_conditions.x_pos -= 0.5
            print(f"Initial X position: {self.initial_conditions.x_pos:.2f}")
            
        elif keycode == ord('5'):  # Increase jump power
            self.jump_controller.jump_height_multiplier += 0.1
            print(f"Jump power: {self.jump_controller.jump_height_multiplier:.2f}")
            
        elif keycode == ord('6'):  # Decrease jump power
            self.jump_controller.jump_height_multiplier = max(0.1, 
                self.jump_controller.jump_height_multiplier - 0.1)
            print(f"Jump power: {self.jump_controller.jump_height_multiplier:.2f}")
            
        elif keycode == ord('7'):  # Add forward velocity
            self.initial_conditions.x_vel += 0.5
            print(f"Initial X velocity: {self.initial_conditions.x_vel:.2f}")
            
        elif keycode == ord('8'):  # Add backward velocity
            self.initial_conditions.x_vel -= 0.5
            print(f"Initial X velocity: {self.initial_conditions.x_vel:.2f}")
            
        elif keycode == ord('9'):  # Add upward velocity
            self.initial_conditions.z_vel += 0.5
            print(f"Initial Z velocity: {self.initial_conditions.z_vel:.2f}")
            
        elif keycode == ord('0'):  # Add downward velocity
            self.initial_conditions.z_vel -= 0.5
            print(f"Initial Z velocity: {self.initial_conditions.z_vel:.2f}")
            
        elif keycode == ord('H'):  # Help
            self.print_help()
    
    def print_help(self):
        """Print help information"""
        print("\n" + "="*50)
        print("KEYBOARD CONTROLS:")
        print("="*50)
        print("SPACE - Pause/Resume simulation")
        print("R     - Reset simulation and start new jump")
        print("N     - Start new jump (without full reset)")
        print("S     - Step once (when paused)")
        print("J     - Toggle automatic jump controller")
        print("P     - Print current state and jump status")
        print("H     - Show this help")
        print()
        print("INITIAL CONDITIONS:")
        print("1/2   - Increase/Decrease initial height")
        print("3/4   - Move initial position forward/backward")
        print("7/8   - Add forward/backward initial velocity")
        print("9/0   - Add upward/downward initial velocity")
        print()
        print("JUMP PARAMETERS:")
        print("5/6   - Increase/Decrease jump power")
        print()
        print("JUMP BEHAVIOR:")
        print("- Robot jumps once, then holds landing position")
        print("- Press 'R' to reset and jump again")
        print("- Press 'N' to start new jump from current position")
        print("="*50)
    
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
    model_path = r"C:/Users/eligi/Downloads/simple_hj_model/src/Mujoco Simulations/yepers3.xml"
    
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