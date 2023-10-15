import mujoco
import os
from gym.envs.mujoco import MujocoEnv
from gym.envs.mujoco.mujoco_rendering import Viewer
from gym.envs.mujoco.mujoco_rendering import RenderContextOffscreen
import time
import numpy as np

model_path = "../../fancy_gym/fancy_gym/envs/mujoco/sb_planning/assets/ur10e_sb_planning.xml"
model_path = os.path.abspath(model_path)
# Test mujoco bindings
model = mujoco.MjModel.from_xml_path("../ur10e_sb_planning.xml")
data = mujoco.MjData(model)
print(data.ctrl)
viewer = Viewer(model, data)

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sbp_wrapper.collision_checker import CollisionChecker
from sbp_wrapper.ompl_planner import OMPLPlanner


urdf_file = "../../ur10e_mujoco/urdf/ur10e_res_rl.urdf"
sdf_file = "../../ur10e_mujoco/urdf/ur10e.srdf"
asset_dir = "../../ur10e_mujoco"

collision_checker = CollisionChecker(urdf_file, sdf_file, asset_dir)
ompl_planner = OMPLPlanner()
start_joint = [0, 0, 0, 0, 0, 0]
goal_joint = [3.14, 0 , 0, 0, 0, 0]

joint_1 = [92, -65, 122, -162, -87, 0]
joint_2 = [144, -34, 62, -162, -87, 0]
joint_3 = [26, -42, 87, -162, -87, 0]


ompl_planner.set_validity_checker(collision_checker.is_state_valid)

ompl_planner.set_planning_request(start_joint, goal_joint)

result = ompl_planner.solve()

position = result["position"]
n_points = len(position)

solution = ompl_planner.get_solution(interpolate=True, num_points=20000)
print("Solution length: ", len(solution))

for joint in solution:
    if collision_checker.check_collision(joint):
        print("Collision detected - before time parameterization")

for joint in position:
    if collision_checker.check_collision(joint):
        print("Collision detected - after time parameterization")
        break

now = time.time()
cnt = 0
while data.time < 100:
    mujoco.mj_step(model, data, nstep=1)
    index = min(cnt, n_points-1)
    data.ctrl = position[index]
    viewer.render()
    cnt += 1
    if collision_checker.check_collision(data.qpos):
        print("Collision detected")


diff = time.time() - now
print("Time taken: ", diff)
