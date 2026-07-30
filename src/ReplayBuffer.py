import os
import pickle
import random
import numpy as np

class ReplayMemory:
    def __init__(self, capacity, seed):
        """
        The Replay Buffer stores past (state, action, reward, next_state, done) tuples
        so we can randomly sample them for training
        :param capacity: maximum number of transitions to store
        :param seed: for reproducible random sampling via Python's random
        """
        random.seed(seed)
        self.capacity = capacity
        self.buffer = [] #A list to hold the tuples
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """push -method for inserting transitions"""
        #Convert data types to np.arrays first
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        action = np.array(action, dtype=np.float32)
        reward = np.array([reward], dtype=np.float32)
        done = np.array([done], dtype=np.float32)

        """Grow until a capacity is met"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        """make sure that when the buffer is full, it discards the oldest information first (FIFO)"""
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """sample -method for random sampling of batches"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        """Methods for saving and loading buffers"""
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity
