import mujoco
import os
from gym.envs.mujoco import MujocoEnv
from gym.envs.mujoco.mujoco_rendering import Viewer
from gym.envs.mujoco.mujoco_rendering import RenderContextOffscreen
import time

model_path = "../ur10e_sb_planning.xml"
model_path = os.path.abspath(model_path)
# Test mujoco bindings
model = mujoco.MjModel.from_xml_path("../ur10e_sb_planning.xml")
data = mujoco.MjData(model)
print(data.ctrl)
viewer = Viewer(model, data)

now = time.time()
while data.time < 4:
    mujoco.mj_step(model, data, nstep=10)
    data.ctrl = [0, -1.5708, 1.5708, -1.5708, -1.5708, 0]
    # data.ctrl = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0]
    viewer.render()
diff = time.time() - now
print("Time taken: ", diff)
