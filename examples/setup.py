import mujoco
import mujoco.viewer as viewer

pit = mujoco.MjSpec.from_file('./src/xmls/pit.xml')
pit.modelname = "pit"

rod = mujoco.MjSpec.from_file('./src/xmls/rod.xml')
rod.modelname = "rod"

multisegment = mujoco.MjSpec.from_file('./src/xmls/multi-segment.xml')

world = pit.attach(rod,frame=pit.frames[0], prefix='child-')
world = pit.attach(multisegment, frame=pit.frames[0], prefix='multi-')

model = pit.compile()