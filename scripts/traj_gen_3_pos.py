import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sbp_wrapper.collision_checker import CollisionChecker
from sbp_wrapper.ompl_planner import OMPLPlanner
import yaml


def from_degrees(jnt):
    return [np.deg2rad(j) for j in jnt]

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

joint_1 = from_degrees(joint_1)
joint_2 = from_degrees(joint_2)
joint_3 = from_degrees(joint_3)

combinations = [(joint_1, joint_2), (joint_2, joint_3), (joint_3, joint_1),
                (joint_2, joint_1), (joint_3, joint_2), (joint_1, joint_3)]


ompl_planner.set_validity_checker(collision_checker.is_state_valid)

cnt = 0
for jnt_1, jnt_2 in combinations:
    ompl_planner.set_planning_request(jnt_1, jnt_2)

    result = ompl_planner.solve()

    position = result["position"]

    for joint in position:
        if collision_checker.check_collision(joint):
            print("Collision detected - after time parameterization")
            break

    position = [pos.tolist() for pos in position]

    yaml_file = "../assets/traj/ait_traj_{}.yaml".format(cnt)
    with open(yaml_file, "w") as f:
        yaml.dump(position, f)
    cnt += 1

