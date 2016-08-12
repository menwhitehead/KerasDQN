from dqn import DQN
from dqn_globals import *
import numpy as np
import sys
import time

class DQNTraining:
  
  def __init__(self, environment, dqn):
    self.environment = environment
    self.dqn = dqn
    self.dqn.compileModel()
    self.past_frames = np.zeros((SEQUENCE_FRAME_COUNT, FRAME_WIDTH, FRAME_HEIGHT))
    self.frame_index = 1

  def playOneEpisode(self, epsilon):
    total_score = 0.0
    episode_frame = 0
    while not self.environment.isEpisodeOver() and episode_frame < MAX_FRAMES_PER_EPISODE:
      episode_frame += 1
      if VERBOSE:
        if self.dqn.iteration % 10 == 0:
          print "frame: ", self.dqn.iteration

      current_frame = self.environment.getScreen();
  
      #self.past_frames.append(current_frame)
      if self.frame_index < SEQUENCE_FRAME_COUNT:
        self.past_frames[self.frame_index] = current_frame
        self.frame_index += 1
        self.environment.performRandomAction()

      elif self.frame_index >= SEQUENCE_FRAME_COUNT:
        #print self.past_frames[1:].shape, current_frame.shape
        self.past_frames = np.vstack((self.past_frames[1:], np.reshape(current_frame, (1, FRAME_WIDTH, FRAME_HEIGHT))))
        #self.past_frames = np.roll(self.past_frames, 1)
        #self.past_frames[-1] = np.reshape(current_frame, (1, FRAME_SIZE, FRAME_SIZE))
        
        action = self.dqn.selectAction(self.past_frames, epsilon)  # send a 0 epsilon
        reward = self.environment.performAction(action)
        total_score += reward;
  
        # # Normalize score to -1, 0, 1
        # if reward != 0:
        #   reward /= abs(reward)
        # 
        # if self.environment.isEpisodeOver():
        #   self.dqn.replay_memory.addTransition(self.past_frames, action, reward, None)
        # else:
        #   self.dqn.replay_memory.addTransition(self.past_frames, action, reward, self.environment.getScreen())
        # 
        # if len(self.dqn.replay_memory) > REPLAY_MEMORY_INIT_SIZE:
        #   self.dqn.update()
          
      #print "PAST FRAMES:", self.past_frames
      #print str(self.dqn.replay_memory)
    self.environment.reset()
    return total_score
  
  

  def test(self, number_episodes, epsilon, model_filename):
    if LOG:
      log_file = open(TESTING_LOG, 'w')
      log_file.write("Episode, Iteration, Score, Time\n" % ())
    
    self.dqn.loadModel(model_filename)  # load the trained model
  
    total_score = 0.0
    total_time = 0.0
    running_average = 0.0
  
    start_time = time.time()
  
    for episode in range(number_episodes):
      test_score = self.playOneEpisode(epsilon)
      total_score += test_score
      print "\t Episode %d Score: %d" % (episode, test_score)   
        
      if LOG:
        log_file.write("%d, %d\n" % (episode, test_score))
        log_file.flush()
        
    print "FINAL AVERAGE SCORE: %.4f" % (total_score / float(number_episodes))



if __name__ == "__main__":
  environment = getEnvironment() 
  dqn = DQN(environment, environment.getActionSet())
  training = DQNTraining(environment, dqn)
  training.test(100, 0.0, sys.argv[1])


