
import random
import numpy as np

class Environment:
    
    def __init__(self):
        self.screen = np.random.rand(84, 84)
    
    def getActionSet(self):
        return [0, 1, 2]
    
    def performAction(self, action):
        #print "performing action", action
        if action == 0:
            return 1.0  # reward
        else:
            return -1.0
    
    def performRandomAction(self):
        action = random.choice(self.getActionSet())
        #print "performing random action", action
        
        return 0.0 # return reward
    
    def reset(self):
        #print "Env reset"
        pass
        
    def getScreen(self):
        # must return a numpy array!
        return self.screen
        
    def episodeOver(self):
        return random.random() < 0.1