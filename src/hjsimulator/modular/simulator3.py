import mujoco
import mujoco.viewer
import numpy as np
import time
from pit_rod import pit_rod_xml
from pit_rod import set_rod_initial_conditions
from pit_humanoid import pit_humanoid_xml
from pit_humanoid import set_humanoid_initial_conditions





class Simulator:
    
    def __init__(self, module: str, callbacks: list = [], nt: int = 1000, params: list = None):
        
        self.module = module
        self.callbacks = callbacks
        self.nt = nt
        self.paused = True

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
    callbacks = []
    simulator = Simulator(module, callbacks, nt = 2000, params = [3, -2, 3, -50, 0, 2, 0, 0])

    # simulator.simulate()
    simulator.visualize()