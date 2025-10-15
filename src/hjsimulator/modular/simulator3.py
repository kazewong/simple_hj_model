import mujoco
import mujoco.viewer
import numpy as np
import time
from pit_rod import pit_rod_xml
from pit_rod import set_rod_initial_conditions
from pit_humanoid import pit_humanoid_xml
from pit_humanoid import set_humanoid_initial_conditions

collision_finished = False

# callback method
def track_forces(model, data, step):
    global collision_finished

    if collision_finished:
        return
    
    elif data.ncon > 0:  # if there are contacts
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, 0, force)  # get first contact force
        with open("C:/Users/eligi/revamp/src/hjsimulator/modular/forces.txt", "a") as f:
            f.write(f"Step {step}: Contact force = {np.linalg.norm(force[:3]):.2f}\n")

    elif data.ncon == 0:
        with open("C:/Users/eligi/revamp/src/hjsimulator/modular/forces.txt", "r") as f:
            if f.read().strip():  # file has content, so contact happened
                collision_finished = True


class Simulator:
    
    def __init__(self, module: str, callbacks: list = [], nt: int = 1000, params: list = None):
        
        self.module = module
        self.callbacks = callbacks
        self.nt = nt
        self.paused = True

        # module options
        if module == 'rod':
            self.model = mujoco.MjModel.from_xml_string(pit_rod_xml)
            self.data = mujoco.MjData(self.model)
            set_rod_initial_conditions(self.model, self.data, params)
        elif module == 'humanoid':
            self.model = mujoco.MjModel.from_xml_string(pit_humanoid_xml)
            self.data = mujoco.MjData(self.model)
            set_humanoid_initial_conditions(self.model, self.data, params)
        else:
            print("Error: Module not found")

    def key_callback(self, keycode):
        if keycode == 32:
            self.paused = not self.paused
            print(f"Simulation {'paused' if self.paused else 'unpaused'}")

    # animate simulation (adding visualize as a callback method makes things more complicated, should we still do it?)
    def visualize(self):
        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_callback) as viewer:
                        
            for i in range(self.nt):
                if not self.paused:
                    mujoco.mj_step(self.model, self.data)
                    
                    for callback in self.callbacks: 
                        callback(self.model, self.data, i)
                
                viewer.sync()
                time.sleep(0.01)


if __name__ == "__main__":
    module = 'humanoid' # rod/humanoid
    callbacks = [track_forces]
    simulator = Simulator(module, callbacks, nt = 2000, params = [10, -5, 2, 45, -30, 7, 0, 0])  # (hv, vv, d, a, b, avm, g, avn)

    # simulator.simulate()
    simulator.visualize()