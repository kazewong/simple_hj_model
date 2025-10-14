from models.pit import PitModel
from models.rod import RodModel
from models.multisegment import MultiSegmentModel

from models.pit import pit_xml
from models.rod import rod_xml
from models.multisegment import multisegment_xml

import mujoco
import mujoco.viewer


def track_forces(modules, data, step):
    # track forces
    return

class Simulator:
    
    def __init__(self, modules: list, callbacks: list = [], nt: int = 1000):
        
        self.modules = modules
        self.callbacks = callbacks
        self.nt = nt

        # Initialize the unified XML string
        unified_xml = "<mujoco model='combined'>"
        
        # Append each module's XML representation
        if "pit" in modules:
            unified_xml += pit_xml.lstrip('<mujoco>').rstrip('</mujoco>')
        if "rod" in modules:
            unified_xml += rod_xml.lstrip('<mujoco>').rstrip('</mujoco>')
        if "humanoid" in modules:
            unified_xml += multisegment_xml.lstrip('<mujoco>').rstrip('</mujoco>')

        unified_xml += "</mujoco>"

        # print(unified_xml)
        with open('C:/Users/eligi/revamp/src/hjsimulator/unified_xml.txt', 'w') as file:
            file.write(unified_xml)

        
        # Create the unified MuJoCo model
        self.model = mujoco.MjModel.from_xml_string(unified_xml)
        self.data = mujoco.MjData(self.model)
    
    def simulate(self, dt: float = 0.1):
        for i in range(self.nt):
            mujoco.mj_step(self.model, self.data)
            for callback in self.callbacks:
                callback(self.model, self.data, i)

                
    def visualize(self):
        mujoco.viewer.launch(self.model, self.data)



if __name__ == "__main__":
    modules = ['pit', 'rod', 'humanoid']
    callbacks = [track_forces]
    simulator = Simulator(modules, callbacks, nt = 1000)

    simulator.simulate(dt = 0.02)
    simulator.visualize()