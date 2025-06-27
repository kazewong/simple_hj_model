import mujoco
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# --- MODEL DEFINITION ---
xml = """
<mujoco>
    <option gravity="0 0 -9.81"/>
    <worldbody>
        <camera name="sideview" pos="4 0 0.5" euler="0 90 90"/>
        <geom name="ground" type="plane" size="2 2 0.1" pos="0 0 0" rgba="0 0.6 0 1" friction="1 0.005 0.0001"/>
        <body name="rodd" pos="0 -0.5 1">
            <freejoint/>
            <geom name="rod" type="cylinder" pos="0 0 0" size="0.05 0.25" rgba="0.2 0.2 0.8 1" solref="0.0001 0.001" solimp="0.99 0.99 0.01" friction="1 0.005 0.0001"/>
        </body>
    </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)

# --- INITIAL STATE SETUP ---
data.qpos[0] = 0      # x
data.qpos[1] = -1.5   # y
data.qpos[2] = 1      # z
angle = np.deg2rad(60)
data.qpos[3] = np.cos(angle / 2)  # w
data.qpos[4] = np.sin(angle / 2)  # x
data.qpos[5] = 0                  # y
data.qpos[6] = 0                  # z

data.qvel[0] = 0.0   # vx
data.qvel[1] = 5.0   # vy
data.qvel[2] = -5.0  # vz
data.qvel[3] = -5.0  # wx
data.qvel[4] = 0.0   # wy
data.qvel[5] = 0.0   # wz

# --- SIMULATION LOOP AND DATA COLLECTION ---
positions = []
linear_velocities = []
angular_velocities = []
external_forces = []
saved_qpos = []
saved_qvel = []

num_steps = 700
for _ in range(num_steps):
    mujoco.mj_step(model, data)
    positions.append(data.xipos[1].copy())
    linear_velocities.append(data.qvel[:3].copy())
    angular_velocities.append(data.qvel[3:].copy())
    saved_qpos.append(data.qpos.copy())
    saved_qvel.append(data.qvel.copy())

    force = np.zeros(6)
    net_force = np.zeros(3)
    for i in range(data.ncon):
        contact = data.contact[i]
        mujoco.mj_contactForce(model, data, i, force)
        if contact.geom1 == model.body_geomadr[1] or contact.geom2 == model.body_geomadr[1]:
            net_force += force[:3]
    external_forces.append(net_force.copy())

# Convert to NumPy arrays
positions = np.array(positions)
linear_velocities = np.array(linear_velocities)
angular_velocities = np.array(angular_velocities)
external_forces = np.array(external_forces)
saved_qpos = np.array(saved_qpos)
saved_qvel = np.array(saved_qvel)

# --- PLOTTING ---

# 1. Position
plt.figure()
plt.plot(positions[:, 0], label='x')
plt.plot(positions[:, 1], label='y')
plt.plot(positions[:, 2], label='z')
plt.xlabel('Time step')
plt.ylabel('Position (m)')
plt.title('Cylinder Position Throughout Simulation')
plt.legend()
plt.grid(True)
plt.show()

# 2. Linear Velocity
plt.figure()
plt.plot(linear_velocities[:, 0], label='vx')
plt.plot(linear_velocities[:, 1], label='vy')
plt.plot(linear_velocities[:, 2], label='vz')
plt.xlabel('Time step')
plt.ylabel('Linear Velocity (m/s)')
plt.title('Cylinder Linear Velocity Over Time')
plt.legend()
plt.grid(True)
plt.show()

# 3. Angular Velocity
plt.figure()
plt.plot(angular_velocities[:, 0], label='wx')
plt.plot(angular_velocities[:, 1], label='wy')
plt.plot(angular_velocities[:, 2], label='wz')
plt.xlabel('Time step')
plt.ylabel('Angular Velocity (rad/s)')
plt.title('Cylinder Angular Velocity Over Time')
plt.legend()
plt.grid(True)
plt.show()

# 4. External Forces
plt.figure()
plt.plot(external_forces[:, 0], label='Fx')
plt.plot(external_forces[:, 1], label='Fy')
plt.plot(external_forces[:, 2], label='Fz')
plt.xlabel('Time step')
plt.ylabel('External Force (N)')
plt.title('External Forces on Rod Over Time')
plt.legend()
plt.grid(True)
plt.show()

# --- ANIMATION USING SAVED STATES ---
frames = []
num_frames = 300  # or any number <= num_steps

# Create a new data object for rendering (optional, for safety)
data_render = mujoco.MjData(model)

for i in range(num_frames):
    data_render.qpos[:] = saved_qpos[i]
    data_render.qvel[:] = saved_qvel[i]
    mujoco.mj_forward(model, data_render)  # update derived quantities
    renderer.update_scene(data_render, camera="sideview")
    frame = renderer.render()
    frames.append(frame)

fig, ax = plt.subplots()
im = ax.imshow(frames[0])
ax.axis('off')

def update(i):
    im.set_data(frames[i])
    return [im]

ani = FuncAnimation(fig, update, frames=num_frames, interval=20, blit=True)
plt.close(fig)
HTML(ani.to_jshtml())