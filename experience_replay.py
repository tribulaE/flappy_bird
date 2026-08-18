from collections import deque
import random


# Replay Memory buffer for storing and sampling transitions from agent experience
class ReplayMemory():

    
    def __init__(self, maxlen, seed=None):
        # Initialize a fixed-size deque to store transitions (state, action, reward, next_state)
        self.memory = deque([], maxlen=maxlen)

        # Optional seed for reproducibility
        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        # Add a new transition to the memory buffer
        self.memory.append(transition)

    def sample(self, sample_size):
        # Randomly sample a batch of transitions from memory for training
        return random.sample(self.memory, sample_size)

    def __len__(self):
        # Return the current number of transitions stored in memory
        return len(self.memory)

