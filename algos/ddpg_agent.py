from .agent_base import BaseAgent
from .ddpg_utils import Policy, Critic, ReplayBuffer, soft_update_params
import utils.common_utils as cu
import torch
import numpy as np
import torch.nn.functional as F
import copy, time
from pathlib import Path
from torch.distributions import MultivariateNormal

def to_numpy(tensor):
    return tensor.cpu().numpy().flatten()

class DDPGAgent(BaseAgent):
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
    

    def update(self,):
        """ After collecting one trajectory, update the pi and q for #transition times: """
        info = {}
        update_iter = self.buffer_ptr - self.buffer_head # update the network once per transition

        if self.buffer_ptr > self.random_transition: # update once we have enough data
            for _ in range(update_iter):
                info = self._update()
        
        # update the buffer_head:
        self.buffer_head = self.buffer_ptr
        return info
    
    @torch.no_grad()
    def get_action(self, observation, evaluation=False):
        # Add the batch dimension
        if observation.ndim == 1:
            observation = observation[None]

        # Convert the observation to torch tensor
        x = torch.from_numpy(observation).float().to(self.device)

        # The stddev of the expl_noise if not evaluation
        expl_noise = 0.05 * self.max_action

        # Get the action
        with torch.no_grad():
            action = self.pi(x).squeeze(0)

        if not evaluation:
            # Collect random trajectories for better exploration.
            if self.buffer_ptr < self.random_transition:
                # Random actions of shape (action_dim,)
                action = torch.FloatTensor(self.action_dim).uniform_(-1, 1)
                return action, {}

            # Create a multivariate normal distribution
            m = MultivariateNormal(
                torch.zeros(action.shape), torch.eye(action.shape[0]) * expl_noise**2
            )
            # Shape of noise: (action_dim,)
            noise = m.sample()
            action += noise

        # Clip the action within the bounds of the action space
        action = action.clamp(-self.max_action, self.max_action)
        return action, {}  # just return a positional value


    # 1. compute target Q, you should not modify the gradient of the variables
    def calculate_target(self, batch):
        ########## Your code starts here. ##########
        with torch.no_grad():
            next_action = self.pi_target(batch.next_state)
            q_tar = self.q_target(batch.next_state, next_action)
            target_Q = batch.reward + batch.not_done*self.gamma*q_tar
        ########## Your code ends here. ##########
        return target_Q
        
    # 2. compute critic loss
    def calculate_critic_loss(self, current_Q, target_Q):
        ########## Your code starts here. ##########
        critic_loss = F.mse_loss(current_Q, target_Q)
        ########## Your code ends here. ##########
        return critic_loss

    # 3. compute actor loss
    def calculate_actor_loss(self, batch):
        ########## Your code starts here. ##########
        actor_loss = -self.q(batch.state, self.pi(batch.state)).mean()
        ########## Your code ends here. ##########
        return actor_loss

    def _update(self,):
        # get batch data
        batch = self.buffer.sample(self.batch_size, device=self.device)
        #    batch contains:
        #    state = batch.state, shape [batch, state_dim]
        #    action = batch.action, shape [batch, action_dim]
        #    next_state = batch.next_state, shape [batch, state_dim]
        #    reward = batch.reward, shape [batch, 1]
        #    not_done = batch.not_done, shape [batch, 1]

        """
        # TODO: Get the current Q estimate
        """
        current_Q = self.q(batch.state, batch.action)

        target_Q = self.calculate_target(batch)
        critic_loss = self.calculate_critic_loss(current_Q, target_Q)

        # optimize the critic
        self.q_optim.zero_grad()
        critic_loss.backward()
        self.q_optim.step()

        actor_loss = self.calculate_actor_loss(batch)
        
        # optimize the actor
        self.pi_optim.zero_grad()
        actor_loss.backward()
        self.pi_optim.step()

        """
        # TODO: update the target q and pi using u.soft_update_params() (See the DQN code)
        """
        soft_update_params(self.q, self.q_target, self.tau)
        soft_update_params(self.pi, self.pi_target, self.tau)

        return {}

    def record(self, state, action, next_state, reward, done):
        """ Save transitions to the buffer. """
        self.buffer_ptr += 1
        self.buffer.add(state, action, next_state, reward, done)

    def train_iteration(self):
        #start = time.perf_counter()
        # Run actual training        
        reward_sum, timesteps, done = 0, 0, False
        # Reset the environment and observe the initial state
        obs, _ = self.env.reset()
        while not done:
            
            # Sample action from policy
            action_tuple = self.get_action(obs)
            action, _ = action_tuple

            # Perform the action on the environment, get new state and reward
            next_obs, reward, done, _, _ = self.env.step(to_numpy(action))

            # Store action's outcome (so that the agent can improve its policy)        
            
            done_bool = float(done) if timesteps < self.max_episode_steps else 0 
            self.record(obs, action, next_obs, reward, done_bool)
                
            # Store total episode reward
            reward_sum += reward
            timesteps += 1
            
            if timesteps >= self.max_episode_steps:
                done = True
            # update observation
            obs = next_obs.copy()

        # update the policy after one episode
        #s = time.perf_counter()
        info = self.update()
        #e = time.perf_counter()
        
        # Return stats of training
        info.update({
                    'episode_length': timesteps,
                    'ep_reward': reward_sum,
                    })
        
        end = time.perf_counter()
        return info
        
    def train(self):
        if self.cfg.save_logging:
            L = cu.Logger() # create a simple logger to record stats
        start = time.perf_counter()
        total_step=0
        run_episode_reward=[]
        log_count=0
        
        for ep in range(self.cfg.train_episodes + 1):
            # collect data and update the policy
            train_info = self.train_iteration()
            train_info.update({'episodes': ep})
            total_step+=train_info['episode_length']
            train_info.update({'total_step': total_step})
            run_episode_reward.append(train_info['ep_reward'])
            
            if total_step>self.cfg.log_interval*log_count:
                average_return=sum(run_episode_reward)/len(run_episode_reward)
                if not self.cfg.silent:
                    print(f"Episode {ep} Step {total_step} finished. Average episode return: {average_return}")
                if self.cfg.save_logging:
                    train_info.update({'average_return':average_return})
                    L.log(**train_info)
                run_episode_reward=[]
                log_count+=1

        if self.cfg.save_model:
            self.save_model()
            
        logging_path = str(self.logging_dir)+'/logs'   
        if self.cfg.save_logging:
            L.save(logging_path, self.seed)
        self.env.close()

        end = time.perf_counter()
        train_time = (end-start)/60
        print('------ Training Finished ------')
        print(f'Total traning time is {train_time}mins')
        
    def load_model(self):
        # define the save path, do not modify
        filepath=str(self.model_dir)+'/model_parameters_'+str(self.seed)+'.pt'
        
        d = torch.load(filepath)
        self.q.load_state_dict(d['q'])
        self.q_target.load_state_dict(d['q_target'])
        self.pi.load_state_dict(d['pi'])
        self.pi_target.load_state_dict(d['pi_target'])
    
    def save_model(self):   
        # define the save path, do not modify
        filepath=str(self.model_dir)+'/model_parameters_'+str(self.seed)+'.pt'
        
        torch.save({
            'q': self.q.state_dict(),
            'q_target': self.q_target.state_dict(),
            'pi': self.pi.state_dict(),
            'pi_target': self.pi_target.state_dict()
        }, filepath)
        print("Saved model to", filepath, "...")
        
        
