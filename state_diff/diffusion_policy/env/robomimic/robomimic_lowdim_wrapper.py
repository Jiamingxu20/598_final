from typing import List, Dict, Optional
import numpy as np
import gym
from gym.spaces import Box
from robomimic.envs.env_robosuite import EnvRobosuite
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.common.robomimic_util import robomimic_process_observations

class RobomimicLowdimWrapper(gym.Env):
    def __init__(self, 
        env: EnvRobosuite,
        obs_keys: List[str]=[
            'object', 
            'robot0_eef_pos', 
            'robot0_eef_quat', 
            'robot0_gripper_qpos'],
        init_state: Optional[np.ndarray]=None,
        render_hw=(256,256),
        render_camera_name='agentview',
        rotation_rep='rotation_6d',
        obj_pos_idx=None,
        obj_quat_idx=None,
        ):

        self.env = env
        self.obs_keys = obs_keys
        self.init_state = init_state
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.seed_state_map = dict()
        self._seed = None
        self.quat_transformer = RotationTransformer(
            from_rep='quaternion', to_rep=rotation_rep)
        
        assert obj_pos_idx is not None and obj_quat_idx is not None, "obj_pos_idx and obj_quat_idx should be provided."

        self.obj_pos_idx = obj_pos_idx
        self.obj_quat_idx = obj_quat_idx

        # setup spaces
        low = np.full(env.action_dimension, fill_value=-1)
        high = np.full(env.action_dimension, fill_value=1)
        self.action_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype
        )
        obs_example = self.get_observation()
        low = np.full_like(obs_example, fill_value=-1)
        high = np.full_like(obs_example, fill_value=1)
        self.observation_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype
        )
        

    def get_observation(self):
        raw_obs = self.env.get_observation()
        # obs = np.concatenate([
        #     self.quat_transformer.forward(raw_obs[key]) if 'quat' in key else raw_obs[key] for key in self.obs_keys 
        # ], axis=0).astype(np.float32)

        obs = robomimic_process_observations(self.obs_keys, raw_obs, self.quat_transformer, self.obj_pos_idx, self.obj_quat_idx, single_obs=True)    
        return obs

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed
    
    def reset(self):
        if self.init_state is not None:
            # always reset to the same state
            # to be compatible with gym
            self.env.reset_to({'states': self.init_state})
        elif self._seed is not None:
            # reset to a specific seed
            seed = self._seed
            if seed in self.seed_state_map:
                # env.reset is expensive, use cache
                self.env.reset_to({'states': self.seed_state_map[seed]})
            else:
                # robosuite's initializes all use numpy global random state
                np.random.seed(seed=seed)
                self.env.reset()
                state = self.env.get_state()['states']
                self.seed_state_map[seed] = state
            self._seed = None
        else:
            # random reset
            self.env.reset()

        # return obs
        obs = self.get_observation()
        return obs
    
    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        # obs = np.concatenate([
        #     self.quat_transformer.forward(raw_obs[key]) if 'quat' in key else raw_obs[key] for key in self.obs_keys 
        # ], axis=0).astype(np.float32)

        obs = robomimic_process_observations(self.obs_keys, raw_obs, self.quat_transformer, self.obj_pos_idx, self.obj_quat_idx, single_obs=True)

        return obs, reward, done, info
    
    def render(self, mode='rgb_array'):
        h, w = self.render_hw
        return self.env.render(mode=mode, 
            height=h, width=w, 
            camera_name=self.render_camera_name)


def test():
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    from matplotlib import pyplot as plt

    dataset_path = '/home/cchi/dev/diffusion_policy/data/robomimic/datasets/square/ph/low_dim.hdf5'
    env_meta = FileUtils.get_env_metadata_from_dataset(
        dataset_path)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=False,
        use_image_obs=False, 
    )
    wrapper = RobomimicLowdimWrapper(
        env=env,
        obs_keys=[
            'object', 
            'robot0_eef_pos', 
            'robot0_eef_quat', 
            'robot0_gripper_qpos'
        ]
    )

    states = list()
    for _ in range(2):
        wrapper.seed(0)
        wrapper.reset()
        states.append(wrapper.env.get_state()['states'])
    assert np.allclose(states[0], states[1])

    img = wrapper.render()
    plt.imshow(img)
    # wrapper.seed()
    # states.append(wrapper.env.get_state()['states'])
