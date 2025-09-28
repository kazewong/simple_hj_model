import mujoco

class MuJoCoSimulator:
    
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)

