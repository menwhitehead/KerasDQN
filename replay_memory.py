
from dqn_globals import *
import numpy as np
import random

class ReplayMemory:
    
    def __init__(self, memory_size=100):
        self.max_size = memory_size
        self.next_index = 0
        self.is_full = False
        self.frames_memories = np.zeros((self.max_size, SEQUENCE_FRAME_COUNT, FRAME_WIDTH, FRAME_HEIGHT))
        self.action_memories = np.zeros((self.max_size,))
        self.reward_memories = np.zeros((self.max_size,))
        self.next_frame_memories = np.zeros((self.max_size, SEQUENCE_FRAME_COUNT, FRAME_WIDTH, FRAME_HEIGHT))

        # Pre-init minibatch arrays
        self.minibatch_frames_memories = np.zeros((MINIBATCH_SIZE, SEQUENCE_FRAME_COUNT, FRAME_WIDTH, FRAME_HEIGHT))
        self.minibatch_action_memories = np.zeros((MINIBATCH_SIZE,))
        self.minibatch_reward_memories = np.zeros((MINIBATCH_SIZE,))
        self.minibatch_next_frame_memories = np.zeros((MINIBATCH_SIZE, SEQUENCE_FRAME_COUNT, FRAME_WIDTH, FRAME_HEIGHT))


    def addTransition(self, input_frames, action, reward, next_frame=None):
        self.frames_memories[self.next_index] = input_frames
        self.action_memories[self.next_index] = action
        self.reward_memories[self.next_index] = reward
        if next_frame != None:
            self.next_frame_memories[self.next_index] = next_frame
        else:
            # Make this negative to signal a NULL value
            self.next_frame_memories[self.next_index][0][0][0] = -1
            #self.next_frame_memories[self.next_index] = None
        self.next_index += 1

        if self.next_index == self.max_size:
            self.is_full = True
            self.next_index = 0
        
        
        
    def pickMinibatch(self):
        for i in range(MINIBATCH_SIZE):
            minibatch_index = random.randrange(len(self))
            self.minibatch_frames_memories[i] = self.frames_memories[minibatch_index]
            self.minibatch_action_memories[i] = self.action_memories[minibatch_index]
            self.minibatch_reward_memories[i] = self.reward_memories[minibatch_index]
            self.minibatch_next_frame_memories[i] = self.next_frame_memories[minibatch_index]
        return self.minibatch_frames_memories, \
                self.minibatch_action_memories, \
                self.minibatch_reward_memories, \
                self.minibatch_next_frame_memories

        
    def pickMinibatch2(self):
        # TODO: SPEED THIS UP SOMEHOW
        minibatch_indexes = np.random.randint(len(self), size=MINIBATCH_SIZE)
        minibatch_frames = self.frames_memories[minibatch_indexes]
        minibatch_actions = self.action_memories[minibatch_indexes]
        minibatch_rewards = self.reward_memories[minibatch_indexes]
        minibatch_next_frames = self.next_frame_memories[minibatch_indexes]
        return minibatch_frames, minibatch_actions, minibatch_rewards, minibatch_next_frames
        
        
    def pickMinibatch3(self):
        minibatch_frames = []
        minibatch_actions = []
        minibatch_rewards = []
        minibatch_next_frames = []
        for i in range(MINIBATCH_SIZE):
            minibatch_index = random.randrange(len(self))
            minibatch_frames.append(self.frames_memories[minibatch_index])
            minibatch_actions.append(self.action_memories[minibatch_index])
            minibatch_rewards.append(self.reward_memories[minibatch_index])
            minibatch_next_frames.append(self.next_frame_memories[minibatch_index])
        return np.stack(minibatch_frames), np.stack(minibatch_actions), np.stack(minibatch_rewards), np.stack(minibatch_next_frames)


    # Fast and NOT random
    def pickMinibatch4(self):
        return (self.frames_memories[:MINIBATCH_SIZE],
                self.action_memories[:MINIBATCH_SIZE],
                self.reward_memories[:MINIBATCH_SIZE],
                self.next_frame_memories[:MINIBATCH_SIZE])
        
        
    def __len__(self):
        if self.is_full:
            return self.max_size
        else:
            return self.next_index
        
        
        
    def __str__(self):
        result = ''
        for i in range(len(self)):
            result += "MEMORY[%d]: \n%s\n " % (i, str(self.frames_memories[i]))
            result += "\t action: %d\n " % (self.action_memories[i])
            result += "\t reward: %d\n " % (self.reward_memories[i])
            result += "\t next: %s\n " % (self.next_frame_memories[i])
        return result
        