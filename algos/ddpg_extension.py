from .agent_base import BaseAgent
from .ddpg_utils import Policy, Critic, ReplayBuffer
from .ddpg_agent import DDPGAgent

import utils.common_utils as cu
import torch
import numpy as np
import torch.nn.functional as F
import copy, time
from pathlib import Path
from collections import deque

def to_numpy(tensor):
    return tensor.cpu().numpy().flatten()

class DDPGExtension(DDPGAgent):

    """
    ddpg base constructor
        def __init__(self, config=None):
        super(DDPGAgent, self).__init__(config)
        self.device = self.cfg.device  # ""cuda" if torch.cuda.is_available() else "cpu"
        self.name = 'ddpg'
        
        self.action_dim = self.action_space_dim
        self.state_dim = self.observation_space_dim
        self.max_action = self.cfg.max_action
        self.lr=self.cfg.lr
        
        self.buffer_size = 1e6
      
        self.pi = Policy(self.state_dim, self.action_dim, self.max_action).to(self.device)
        self.pi_target = copy.deepcopy(self.pi)
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=float(self.lr))

        self.q = Critic(self.state_dim, self.action_dim).to(self.device)
        self.q_target = copy.deepcopy(self.q)
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=float(self.lr))
        
        self.buffer = ReplayBuffer((self.state_dim, ), self.action_dim, max_size=int(float(self.buffer_size)))
        
        self.batch_size = self.cfg.batch_size
        self.gamma = self.cfg.gamma
        self.tau = self.cfg.tau
        
        # used to count number of transitions in a trajectory
        self.buffer_ptr = 0
        self.buffer_head = 0 
        self.random_transition = 5000 # collect 5k random data for better exploration
        self.max_episode_steps=self.cfg.max_episode_steps
    
    """

    def __init__(self, config=None):
        super(DDPGExtension, self).__init__(config)
        # LNSS-specific parameters
        self.n_steps = config.get("n_steps", 20)  # Horizon for N-step reward aggregation
        self.exp_buffer = deque()  # Temporary buffer for N-step transitions

    def store_lnss_transition(self, state, action, reward, next_state, done):
        """
        Store transitions for LNSS in a temporary buffer and process them for N-step updates.
        """
        self.exp_buffer.append((state, action, reward, next_state, done))
        
        # Process the buffer when it has enough transitions or when the episode ends
        if len(self.exp_buffer) >= self.n_steps or done:
            self._process_lnss_buffer(done)

    def _process_lnss_buffer(self, done):
        """
        Compute N-step rewards for LNSS and store transitions in the main replay buffer.
        """
        gamma = self.gamma
        discounted_reward = 0

        # If the buffer contains enough transitions
        while len(self.exp_buffer) >= self.n_steps or (done and len(self.exp_buffer) > 0):
            if len(self.exp_buffer) >= self.n_steps:
                _, _, _, next_state_1, done_1 = self.exp_buffer[self.n_steps - 1]
            else:
                _, _, _, next_state_1, done_1 = self.exp_buffer[-1]

            # Get the earliest transition
            state_0, action_0, reward_0, _, _ = self.exp_buffer.popleft()
            discounted_reward = reward_0
            gamma_acc = gamma

            # Compute N-step discounted reward
            for _, _, r_i, _, _ in self.exp_buffer:
                discounted_reward += r_i * gamma_acc
                gamma_acc *= gamma

            # Apply discount scaling factor
            ds_factor = (gamma - 1) / (gamma_acc - 1)
            discounted_reward *= ds_factor

            # Store transition in the main replay buffer #    def add(self, state, action, next_state, reward, done, extra:dict=None):
            self.record(state_0, action_0, next_state_1, discounted_reward, done_1)

        # Clear the buffer after processing all transitions at the end of the episode
        if done:
            self.exp_buffer.clear()

    def train_iteration(self):
        """
        Train the agent using LNSS transitions.
        """
        reward_sum, timesteps, done = 0, 0, False
        obs, _ = self.env.reset()

        while not done:
            # Select action based on the current policy
            action, _ = self.get_action(obs)
            next_obs, reward, done, _, _ = self.env.step(action.cpu().numpy())

            # Store the transition using LNSS logic
            self.store_lnss_transition(obs, action, reward, next_obs, done)

            # Update observation and accumulate rewards
            obs = next_obs
            reward_sum += reward
            timesteps += 1

            if timesteps >= self.max_episode_steps:
                done = True

        # Perform updates based on the replay buffer
        info = self.update()
        info.update({"episode_length": timesteps, "ep_reward": reward_sum})
        return info