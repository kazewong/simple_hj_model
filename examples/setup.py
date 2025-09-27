import mujoco
import mujoco.viewer as viewer

pit = mujoco.MjSpec.from_file('./src/xmls/pit.xml')
pit.modelname = "pit"

rod = mujoco.MjSpec.from_file('./src/xmls/rod.xml')

world = pit.attach(rod,frame=pit.frames[0], prefix='child-')

model = pit.compile()