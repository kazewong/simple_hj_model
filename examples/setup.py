import mujoco
import mujoco.viewer as viewer
from hjsimulator.models.pit import PitModel
from hjsimulator.models.rod import RodModel
from hjsimulator.models.multisegment import MultiSegmentModel

pit = PitModel()
rod = RodModel()
multisegment = MultiSegmentModel()

world = pit.spec.attach(rod.spec,frame=pit.spec.frames[0], prefix='child-')
world = pit.spec.attach(multisegment.spec, frame=pit.spec.frames[0], prefix='multi-')

pit.spec.bodies[1].geoms[0].size[1] = 2

model = pit.spec.compile()
viewer.launch(model)