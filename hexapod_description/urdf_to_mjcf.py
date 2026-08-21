import mujoco
import os

model = mujoco.MjModel.from_xml_path('./urdf/hexapod_model.urdf')
mujoco.mj_saveLastXML("./mjcf/hexapod_model.xml", model)

print("export to hexapod_model.xml")
