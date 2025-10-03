class Simulator:
    
    def __init__(self, models: list[Mujoco.Spec], callbacks: list = [], nt: int):
        # Create a world with all the mujoco models
    
    def simulate(self, dt: float = 0.1):
        for i in range(nt):
            mujoco.mj_step(self.model, self.data)
            for callback in self.callbacks:
                callback(self.model, self.data, i)
                
    def visualize(self):
        viewer.launch(self.model, self.data)