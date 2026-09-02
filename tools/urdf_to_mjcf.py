import mujoco
import os

model = mujoco.MjModel.from_xml_path('./urdf/snake_robot.urdf')
mujoco.mj_saveLastXML("./mjcf/snake_robot.xml", model)

print("export to snake_robot.xml")
