import mujoco
import mujoco.viewer as viewer
from hjsimulator.models.pit import PitModel
from hjsimulator.models.rod import RodModel
from hjsimulator.models.multisegment import MultiSegmentModel
import numpy as np

pit = PitModel()
rod = RodModel(length=1.0, euler_angle=np.array([-45., 45., 0.]), position=np.array([4., 2.0, 0.02]))
multisegment = MultiSegmentModel()

world = pit.spec.attach(rod.spec,frame=pit.spec.frames[0], prefix='child-')
world = pit.spec.attach(multisegment.spec, frame=pit.spec.frames[0], prefix='multi-')


model = pit.spec.compile()
data = mujoco.MjData(model)

data.qvel[model.joint('child-rod_joint').dofadr[0]:model.joint('child-rod_joint').dofadr[0]+6] = np.array([0., -4., -2., 0., -2., 0.])  # Set initial velocity of the rod to zero

print(data.qvel)
# viewer = mujoco.viewer.launch_passive(model, data)
# viewer.cam.distance = 15.0  # Distance from the camera to the lookat point
# viewer.cam.azimuth = -10    # Azimuth angle in degrees
# viewer.cam.elevation = -5.5 # Elevation angle in degrees

viewer.launch(model, data)