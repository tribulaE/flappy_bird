import random
import torch
from torch import nn
import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from datetime import datetime, timedelta
import argparse


DATE_FORMAT = "%Y-%m-%d %H:%M:%S"  # Format for timestamps in logs


# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

#Save the file as a image
matplotlib.use('Agg')  


#See if we can use GPU for processing, if not, use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'  # Force CPU usage for now
print(f"Using device: {device}")


#Deep Q-Learning Agent
class Agent:

    def __init__(self, hyperparameters_set):

        # Load hyperparameters from the YAML file
        with open('flappy_bird/hp.yml', 'r') as file:
            self_hyperparameters = yaml.safe_load(file)
            hyperparameters = self_hyperparameters[hyperparameters_set]

        self.hyperparameters_set = hyperparameters_set

        # Extract hyperparameters for the specified environment
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.network_sync_rate  = hyperparameters['network_sync_rate']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.discount_factor_g = hyperparameters['discount_factor_g']
        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.fc1_nodes = hyperparameters['fc1_nodes'] #Stop training after reaching this number of rewards

        self.env_make_params = hyperparameters.get('env_make_params', {})

        #Loss fucntion and Optimizer for nn
        self.loss_fn = nn.MSELoss()
        self.optimzer = None

        #Path to run info
        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameters_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameters_set}.pt")
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameters_set}.png")

    def run(self, is_training=True, render=False):

        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting"
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        # Create the Flappy Bird environment
        #env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        reward_per_episode = []
    

        #Create policy and target network (nodes in the hidden layer can be adjusted in the hyperparameters file)
        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)

        if is_training:

            #Initialize epsilon
            epsilon = self.epsilon_init

            #Initialize replay memory
            memory = ReplayMemory(self.replay_memory_size)

            epsilon = self.epsilon_init

            target_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            #Keep track of epsilon decay
            epsilon_history = []

            #Tracking number of steps taken
            step_count = 0

            #Policy network optimizer
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

            #Tracking the best reward achieved so far
            best_reward = -9999999

        else:
            #Load learning policy 
            #Then switch model to evaluation mode
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

            

        for episode in itertools.count():
            state, _ = env.reset()
            #converting to a tensor
            state = torch.tensor(state, dtype=torch.float, device=device)


            terminated = False
            episode_reward = 0.0


            #Perform actions until the episode is terminated or the reward exceeds the stop_on_reward threshold
            while (not terminated and episode_reward < self.stop_on_reward):

                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        #converting state to a two dimensional matrix
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Processing:
                new_state, reward, terminated, truncated, info = env.step(action.item())

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

            #Save ,pdel when new best reward is obtained
            if is_training:
                if episode_reward > best_reward: 
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

            epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
            epsilon_history.append(epsilon)

            if datetime.now() - last_graph_update_time > timedelta(seconds=10):
                self.save_graph(reward_per_episode, epsilon_history)
                last_graph_update_time = start_time

            #If enough experience has been collected
            if len(memory) > self.mini_batch_size:

                #Sample from memory 
                mini_batch = memory.sample(self.mini_batch_size)

                self.optimze(mini_batch, policy_dqn, target_dqn)

                #Copying policy network to target newtork after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0


    def save_graph(self, reward_per_episode, epsilon_history):

            #Save plots
            fig = plt.figure(1)

            #Plot average rewards (Y-axis) per episode (X-axis)
            mean_rewards = np.zeros(len(reward_per_episode))
            for x in range(len(mean_rewards)):
                mean_rewards[x] = np.mean(reward_per_episode[max(0, x-100):(x+1)])

            #Plot on a 1 row x 2 col grid
            plt.subplot(121)
            plt.ylabel("Mean Rewards")
            plt.plot(mean_rewards)

            #Ploting epsilon decay (Y-axis) vs episodes (X-axis)
            plt.subplot(122)
            plt.ylabel("Epsilon Decay")
            plt.plot(epsilon_history)

            plt.subplots_adjust(wspace=1.0, hspace=1.0)


            #save plots
            fig.savefig(self.GRAPH_FILE)
            plt.close(fig)





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
            self.optimizer.zero_grad() #Clear gradients
            loss.backward() #Compute gradients (backprogation)
            self.optimizer.step() #Update network paramters like the weights and bias

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train or test model")
    parser.add_argument('hyperparameters', help="")
    parser.add_argument("--train", action="store_true", help="Train the model")
    args = parser.parse_args()


    dql = Agent(hyperparameters_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)