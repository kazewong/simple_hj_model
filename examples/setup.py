import mujoco
import mujoco.viewer as viewer
from src.models.pit import PitModel
from src.models.rod import RodModel
from src.models.multisegment import MultiSegmentModel

pit = PitModel()
rod = RodModel()
multisegment = MultiSegmentModel()

world = pit.model.attach(rod,frame=pit.model.frames[0], prefix='child-')
world = pit.model.attach(multisegment, frame=pit.model.frames[0], prefix='multi-')

model = pit.model.compile()
viewer.launch(model)