import os

import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import MujocoEnv
from fancy_gym.envs.mujoco.box_pushing.box_pushing_utils import rot_to_quat, get_quaternion_error, rotation_distance
from fancy_gym.envs.mujoco.box_pushing.box_pushing_utils import q_max, q_min, q_dot_max, q_torque_max
from fancy_gym.envs.mujoco.box_pushing.box_pushing_utils import desired_rod_quat

import mujoco

MAX_EPISODE_STEPS_BOX_PUSHING = 100

BOX_POS_BOUND = np.array([[0.3, -0.45, -0.01], [0.6, 0.45, -0.01]])

class BoxPushingEnvBase(MujocoEnv, utils.EzPickle):
    """
    franka box pushing environment
    action space:
        normalized joints torque * 7 , range [-1, 1]
    observation space:

    rewards:
    1. dense reward
    2. time-depend sparse reward
    3. time-spatial-depend sparse reward
    """

    def __init__(self, frame_skip: int = 10):
        utils.EzPickle.__init__(**locals())
        self._steps = 0
        self.init_qpos_box_pushing = np.array([0., 0., 0., -1.5, 0., 1.5, 0., 0., 0., 0.6, 0.45, 0.0, 1., 0., 0., 0.])
        self.init_qvel_box_pushing = np.zeros(15)
        self.frame_skip = frame_skip

        self._q_max = q_max
        self._q_min = q_min
        self._q_dot_max = q_dot_max
        self._desired_rod_quat = desired_rod_quat

        self._episode_energy = 0.
        MujocoEnv.__init__(self,
                           model_path=os.path.join(os.path.dirname(__file__), "assets", "box_pushing.xml"),
                           frame_skip=self.frame_skip,
                           mujoco_bindings="mujoco")
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,))

    def step(self, action):
        action = 10 * np.clip(action, self.action_space.low, self.action_space.high)
        resultant_action = np.clip(action + self.data.qfrc_bias[:7].copy(), -q_torque_max, q_torque_max)

        unstable_simulation = False

        try:
            self.do_simulation(resultant_action, self.frame_skip)
        except Exception as e:
            print(e)
            unstable_simulation = True

        self._steps += 1
        self._episode_energy += np.sum(np.square(action))

        episode_end = True if self._steps >= MAX_EPISODE_STEPS_BOX_PUSHING else False

        box_pos = self.data.body("box_0").xpos.copy()
        box_quat = self.data.body("box_0").xquat.copy()
        target_pos = self.data.body("replan_target_pos").xpos.copy()
        target_quat = self.data.body("replan_target_pos").xquat.copy()
        rod_tip_pos = self.data.site("rod_tip").xpos.copy()
        rod_quat = self.data.body("push_rod").xquat.copy()
        qpos = self.data.qpos[:6].copy()
        qvel = self.data.qvel[:6].copy()

        if not unstable_simulation:
            reward = self._get_reward(episode_end, box_pos, box_quat, target_pos, target_quat,
                                      rod_tip_pos, rod_quat, qpos, qvel, action)
        else:
            reward = -50

        obs = self._get_obs()
        box_goal_pos_dist = 0. if not episode_end else np.linalg.norm(box_pos - target_pos)
        box_goal_quat_dist = 0. if not episode_end else rotation_distance(box_quat, target_quat)
        infos = {
            'episode_end': episode_end,
            'box_goal_pos_dist': box_goal_pos_dist,
            'box_goal_rot_dist': box_goal_quat_dist,
            'episode_energy': 0. if not episode_end else self._episode_energy,
            'is_success': True if episode_end and box_goal_pos_dist < 0.05 and box_goal_quat_dist < 0.5 else False,
            'num_steps': self._steps
        }
        return obs, reward, episode_end, infos

    def reset_model(self):
        # rest box to initial position
        self.set_state(self.init_qpos_box_pushing, self.init_qvel_box_pushing)
        box_init_pos = np.array([0.4, 0.3, -0.01, 0.0, 0.0, 0.0, 1.0])
        self.data.joint("box_joint").qpos = box_init_pos

        # set target position
        box_target_pos = self.sample_context()
        while np.linalg.norm(box_target_pos[:2] - box_init_pos[:2]) < 0.3:
            box_target_pos = self.sample_context()
        # box_target_pos[0] = 0.4
        # box_target_pos[1] = -0.3
        # box_target_pos[-4:] = np.array([0.0, 0.0, 0.0, 1.0])
        self.model.body_pos[2] = box_target_pos[:3]
        self.model.body_quat[2] = box_target_pos[-4:]
        self.model.body_pos[3] = box_target_pos[:3]
        self.model.body_quat[3] = box_target_pos[-4:]

        # set the robot to the right configuration (rod tip in the box)
        desired_tcp_pos = box_init_pos[:3] + np.array([0.0, 0.0, 0.15])
        desired_tcp_quat = np.array([0, 1, 0, 0])
        desired_joint_pos = self.calculateOfflineIK(desired_tcp_pos, desired_tcp_quat)
        self.data.qpos[:7] = desired_joint_pos

        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        self._episode_energy = 0.

        return self._get_obs()

    def sample_context(self):
        # TODO: implement this
        context = None
        return context

    def _get_reward(self, episode_end, box_pos, box_quat, target_pos, target_quat,
                    rod_tip_pos, rod_quat, qpos, qvel, action):
        raise NotImplementedError

    def _get_obs(self):
        # TODO: implement this
        obs = None
        return obs

    def _joint_limit_violate_penalty(self, qpos, qvel, enable_pos_limit=False, enable_vel_limit=False):
        penalty = 0.
        p_coeff = 1.
        v_coeff = 1.
        # q_limit
        if enable_pos_limit:
            higher_error = qpos - self._q_max
            lower_error = self._q_min - qpos
            penalty -= p_coeff * (abs(np.sum(higher_error[qpos > self._q_max])) +
                                  abs(np.sum(lower_error[qpos < self._q_min])))
        # q_dot_limit
        if enable_vel_limit:
            q_dot_error = abs(qvel) - abs(self._q_dot_max)
            penalty -= v_coeff * abs(np.sum(q_dot_error[q_dot_error > 0.]))
        return penalty

    def get_body_jacp(self, name):
        id = mujoco.mj_name2id(self.model, 1, name)
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, None, id)
        return jacp

    def get_body_jacr(self, name):
        id = mujoco.mj_name2id(self.model, 1, name)
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, None, jacr, id)
        return jacr

    def calculateOfflineIK(self, desired_cart_pos, desired_cart_quat):
        """
        calculate offline inverse kinematics for UR10e
        :param desired_cart_pos: desired cartesian position of tool center point
        :param desired_cart_quat: desired cartesian quaternion of tool center point
        :return: joint angles
        """

        # TODO: Implement this function
        q = np.zeros(6)

        return q

