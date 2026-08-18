import random
import torch
from torch import nn
import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml

#See if we can use GPU for processing, if not, use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Agent:

    def __init__(self, hyperparameters_set):
        # Load hyperparameters from the YAML file
        with open('hp.yml', 'r') as file:
            self_hyperparameters = yaml.safe_load(file)
            hyperparamters = self_hyperparameters[hyperparameters_set]

        # Extract hyperparameters for the specified environment
        self.replay_memory_size = hyperparamters['replay_memory_size']
        self.mini_batch_size = hyperparamters['mini_batch_size']
        self.epsilon_start = hyperparamters['epsilon_start']
        self.epsilon_decay = hyperparamters['epsilon_decay']
        self.epsilon_min = hyperparamters['epsilon_min']
        self.learning_rate_a = hyperparamters['learning_rate_a']
        self.discount_factor_g = hyperparamters['discount_factor_g']

        #Loss fucntion and Optimizer for nn
        self.loss_fn = nn.MSELoss()
        self.optimzer = None


    def run(self, is_training=True, render=False):
        # Create the Flappy Bird environment
        #env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        reward_per_episode = []
        epsilon_history = []

        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)

            epsilon = self.epsilon_start

            target_dqn = DQN(num_states, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            #Tracking number of steps taken
            step_count = 0

            #Policy network optimizer
            self.optimzer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
            

        for episode in itertools.count():
            state, _ = env.reset()
            #converting to a tensor
            state = torch.tensor(state, dtype=torch.float32, device=device)


            terminated = False
            episode_reward = 0.0



            while not terminated:

                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        #converting state to a two dimensional matrix
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Processing:
                new_state, reward, terminated, _, info = env.step(action.item())

                #Accumulate reward
                episode_reward += reward

                #Convert new state and reward to tensors on device
                new_state = torch.tensor(new_state, dtype=torch.float32, device=device)
                reward = torch.tensor(reward, dtype=torch.float32, device=device)

                if is_training:
                    memory.append((state, action, reward, new_state, terminated))

                    step_count += 1

                #Move to a new state
                state = new_state

        #Keeping track of the rewards collected per episode
        reward_per_episode.append(episode_reward)

        epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
        epsilon_history.append(epsilon)

        #If enough experience has been collected
        if len(memory) > self.mini_batch_size:

            #Sample from memory 
            mini_batch = memory.sample(self.mini_batch_size)

            self.optimze(mini_batch, policy_dqn, target_dqn)

            #Copying policy network to target newtork after a certain number of steps
            if step_count > self.network_sync_rate:
                target_dqn.load_state_dict(policy_dqn.state_dict())
                step_count = 0
    def optimze(self, mini_batch, policy_dqn, target_dqn):

            #Transpose the list of experiences and separate each element
            states, actions, rewards, next_states, terminations = zip(*mini_batch)

            #Stack tensors to create batch tensors
            states = torch.stack(states)

            actions = torch.stack(actions)

            rewards = torch.stack(rewards)

            next_states = torch.stack(next_states)
            terminations = torch.tensor(terminations).float().to(device)

            with torch.no_grad():
                
                #Calculate target Q values 
                target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(next_states).max(dim=1).values

            #Calculate q values from current policy
            current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

            #Compute loss for the whole minibatch
            loss = self.loss_fn(current_q, target_q)

            #Optimize the model
            self.optimzer.zero_grad() #Clear gradients
            loss.backward() #Compute gradients (backprogation)
            self.optimzer.step() #Update network paramters like the weights and bias

if __name__ == "__main__":
    agent = Agent("cartpole1")
    agent.run(is_training=True, render=True)