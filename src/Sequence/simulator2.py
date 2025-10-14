import mujoco
import mujoco.viewer
import numpy as np
import time
from pit_rod import pit_rod_xml
from pit_rod import set_rod_initial_conditions
from pit_humanoid import set_humanoid_initial_conditions

paused = True

def track_forces(module, data, step):
    # track forces
    return


class Simulator:
    
    def __init__(self, module: str, callbacks: list = [], nt: int = 1000):
        
        self.module = module
        self.callbacks = callbacks
        self.nt = nt

        if module == 'rod':
            self.model = mujoco.MjModel.from_xml_string(pit_rod_xml)
            self.data = mujoco.MjData(self.model)
            set_rod_initial_conditions(self.model, self.data, 3, -1, 1, 45, 0, 0, 0, 0)
        else:
            self.model = mujoco.MjModel.from_xml_path("C:/Users/eligi/revamp/src/hjsimulator/modular/pit_humanoid.xml")
            self.data = mujoco.MjData(self.model)
            set_humanoid_initial_conditions(self.model, self.data, 3, -1, 1, 45, 0, 0, 0, 0)

    
    # def simulate(self):
    #     for i in range(self.nt):
    #         mujoco.mj_step(self.model, self.data)
    #         for callback in self.callbacks: 
    #                 callback(self.model, self.data, i)


    def visualize(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            
            for i in range(self.nt):
                mujoco.mj_step(self.model, self.data)
                
                viewer.sync()
                time.sleep(0.01)



if __name__ == "__main__":
    module = 'rod' # as opposed to 'humanoid'
    callbacks = []
    simulator = Simulator(module, callbacks, nt = 2000)

    # simulator.simulate()
    simulator.visualize()