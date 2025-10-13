from models.pit import PitModel
from models.rod import RodModel
from models.multisegment import MultiSegmentModel

from models.pit import pit_xml
from models.rod import rod_xml
from models.multisegment import multisegment_xml

import mujoco
import mujoco.viewer


def track_forces(model, data, step):
    # track forces
    return

class Simulator:
    
    def __init__(self, models: list, callbacks: list = [], nt: int = 1000):
        
        self.models = models
        self.callbacks = callbacks
        self.nt = nt

        # Initialize the unified XML string
        unified_xml = ""
        
        # Append each model's XML representation
        if any(isinstance(model, PitModel) for model in models):
            unified_xml += pit_xml
        if any(isinstance(model, RodModel) for model in models):
            unified_xml += rod_xml
        if any(isinstance(model, MultiSegmentModel) for model in models):
            unified_xml += multisegment_xml
        
        print(unified_xml)

        
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
    pit = PitModel()
    rod = RodModel()
    humanoid = MultiSegmentModel()
    models = [pit, rod, humanoid]
    callbacks = [track_forces]
    simulator = Simulator(models, callbacks, nt = 1000)

    simulator.simulate(dt = 0.02)
    simulator.visualize()