import mujoco

pit_xml = """
<mujoco model="world">
    <compiler angle="radian" coordinate="local" inertiafromgeom="true" />

    <option timestep="0.002" gravity="0 0 -9.81">
    <flag warmstart="enable" />
    </option>
    <visual>
        <rgba haze="0.15 0.25 0.35 1" />
        <quality shadowsize="2048" />
        <map stiffness="700" shadowscale="0.5" fogstart="10" fogend="15" />
        <global azimuth="-10" elevation="-5.5" />
    </visual>
    <default>
    <geom
            contype="1"
            conaffinity="1"
            friction="3 3 3"
            solimp="0.995 0.995 0.001"
            solref="0.005 0.3"
        />
    </default>

  <worldbody>

  <frame pos="0 0 0">
    <!-- Checkered ground plane -->
    <geom
                name="ground"
                type="plane"
                size="20 20 0.1"
                material="grid"
                contype="1"
                conaffinity="1"
                friction="1.0 0.1 0.05"
            />

    <!-- Lighting setup -->
    <light pos="0 4 6" dir="0 -1 -1" diffuse="0.8 0.8 0.8" />
    <light pos="3 -2 4" dir="-1 1 -1" diffuse="0.4 0.4 0.6" />

    <!-- Red horizontal cylindrical bar at 2m height, 5m long
    <geom name="horizontal_bar" type="cylinder" size="0.05 2.5"
          pos="0 0 2" rgba="1 0 0 1" euler="0 1.5708 0"/> -->

<!-- Detection zone on far side of wall (behind the wall from rod's perspective) -->
      <geom
                name="far_side_detector"
                type="box"
                size="5 2 0.5"
                pos="0 -2 0.005"
                rgba="0 1 0 0.3"
                contype="1"
                conaffinity="1"
            />

<!-- Detection zone on wall -->
      <geom
                name="side_detector"
                type="box"
                size="5 0.05 1.5"
                pos="0 -0.08 0"
                rgba="0 1 0 0.3"
                contype="1"
                conaffinity="1"
            />

    Red X-axis (centered at origin, extends both ways)
    <geom
                name="x_axis"
                type="cylinder"
                size="0.02 10"
                pos="0 0 0.01"
                rgba="1 0 0 1"
                euler="0 1.5708 0"
                contype="0"
                conaffinity="0"
            />

    1.5m high rectangular wall
    <geom
                name="wall"
                type="box"
                size="5 0.03 1.15"
                pos="0 0 1.15"
                rgba="0.8 0.4 0.2 1"
            />

    <!-- Green Y-axis (centered at origin, extends both ways) -->
    <geom
                name="y_axis"
                type="cylinder"
                size="0.02 10"
                pos="0 0 0.01"
                rgba="0 1 0 1"
                euler="1.5708 0 0"
                contype="0"
                conaffinity="0"
            />


        </frame>
  </worldbody>

  <asset>
    <!-- Checkered texture with bigger squares -->
    <texture
            name="checker_texture"
            type="2d"
            builtin="checker"
            rgb1="0.3 0.3 0.3"
            rgb2="0.5 0.5 0.5"
            width="512"
            height="512"
        />

    <!-- Material with bigger squares -->
    <material
            name="checkerboard"
            texture="checker_texture"
            texrepeat="2 2"
            texuniform="true"
            reflectance="0.1"
        />
        <texture
            type="skybox"
            builtin="gradient"
            rgb1="0.4 0.6 0.8"
            rgb2="0.1 0.1 0.2"
            width="512"
            height="512"
        />
        <texture
            name="grid"
            type="2d"
            builtin="checker"
            rgb1="0.2 0.3 0.4"
            rgb2="0.1 0.2 0.3"
            width="512"
            height="512"
            mark="cross"
            markrgb="1 1 1"
        />
        <material
            name="grid"
            texture="grid"
            texrepeat="2 2"
            texuniform="true"
            reflectance="0.2"
        />
        <material
            name="body"
            rgba="0.8 0.6 0.4 1"
            specular="0.3"
            shininess="0.1"
        />
        <material name="foot" rgba="0.2 0.2 0.2 1" specular="0.1" />
        <material name="target" rgba="1 0.2 0.2 0.6" />
        <material name="landing" rgba="0.2 1 0.2 0.4" />
  </asset>

  <visual>
      <global offwidth="1280" offheight="720" />
  </visual>

  <statistic extent="10" center="0 0 1" />
</mujoco>
"""


class PitModel:
    def __init__(self):
        self.spec = mujoco.MjSpec.from_string(pit_xml)
