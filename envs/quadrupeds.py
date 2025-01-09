import mujoco
import mujoco.msh2obj_test
import mujoco.viewer
import numpy as np
import pdb

class UnitreeQuadruped:
    def __init__(self, model):
        self.model = mujoco.MjModel.from_xml_path(model)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model)

        # Initialize the state of the quadruped
        self.reset()

    def reset(self):
        # Reset the state of the quadruped
        mujoco.mj_resetData(self.model, self.data)


    def step(self, action):
        # Apply the action to the quadruped and update the state
        mujoco.mj_step(self.model, self.data)
        pdb.set_trace()
        
        self.state += action  # Simplified state update
        reward = self._compute_reward()
        done = self._check_done()
        info = {}
        return self.state, reward, done, info

    def _compute_reward(self):
        # Compute the reward for the current state
        reward = -np.sum(np.square(self.state))  # Example reward function
        return reward

    def _check_done(self):
        # Check if the episode is done
        done = np.linalg.norm(self.state) > 10  # Example condition
        return done
    


env = UnitreeQuadruped('mujoco_menagerie/unitree_go1/scene.xml')

env.step(None)