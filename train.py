from dqn import DQN
from double_dqn import DoubleDQN
from dqn_globals import *
import numpy as np
import time

class DQNTraining:

  def __init__(self, environment, dqn):
    self.epsilon = START_EPSILON
    self.environment = environment
    self.dqn = dqn
    self.dqn.compileModel()
    self.past_frames = np.zeros((SEQUENCE_FRAME_COUNT, FRAME_WIDTH, FRAME_HEIGHT))
    self.frame_index = 0

  def setEpsilon(self):
    if self.dqn.iteration < EXPLORE:
      self.epsilon = START_EPSILON - (START_EPSILON - END_EPSILON) * (float(self.dqn.iteration) / EXPLORE)
    else:
      self.epsilon = END_EPSILON

  def playOneEpisode(self):
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

        action = self.dqn.selectAction(self.past_frames, self.epsilon)
        reward = self.environment.performAction(action)
        total_score += reward;

        # Normalize score to -1, 0, 1
        if reward != 0:
          reward /= abs(reward)

        if self.environment.isEpisodeOver():
          self.dqn.replay_memory.addTransition(self.past_frames, action, reward, None)
        else:
          self.dqn.replay_memory.addTransition(self.past_frames, action, reward, self.environment.getScreen())

        if len(self.dqn.replay_memory) > REPLAY_MEMORY_INIT_SIZE:
          self.dqn.update()

      #print "PAST FRAMES:", self.past_frames
      #print str(self.dqn.replay_memory)
    self.environment.reset()
    return total_score



  def train(self):
    log_file = open(TRAINING_LOG, 'w')
    log_file.write("Episode, Iteration, Score, Time\n" % ())

    if LOAD_EXISTING and MODEL_FILENAME != '':
      self.dqn.loadModel(MODEL_FILENAME)
      self.dqn.compileModel()

    total_score = 0.0
    total_time = 0.0
    running_average = 0.0

    start_time = time.time()

    for episode in range(MAX_EPISODES):
      episode_start_time = time.time()
      itr_start = self.dqn.iteration
      self.setEpsilon()
      train_score = self.playOneEpisode()
      itr_end = self.dqn.iteration
      total_score += train_score
      episode_end_time = time.time()
      episode_iters = itr_end - itr_start
      episode_time = episode_end_time - episode_start_time
      print "\t Episode %d Score: %d" % (episode, train_score)

      if episode % DISPLAY_FREQUENCY == 0:
        print "*" * 40
        print "Episode %d (%d iterations)" % (episode, dqn.iteration)
        print "\t Time:  %.4f seconds" % episode_time
        itrs_per_sec = episode_iters / episode_time
        print "\t Speed: %.2f iterations/sec" % itrs_per_sec
        if itrs_per_sec != 0:
          time_per_mill = 1e6 / (itrs_per_sec * 60 * 60.)
          print "\t 1mill in %.2f hours" % (time_per_mill)
        print "*" * 40
        #self.dqn.printModel()

      if episode % MODEL_SAVE_FREQUENCY == 0:
        self.dqn.saveModel(MODEL_FILENAME + "A", MODEL_FILENAME + "B")
        # self.dqn.saveModel(MODEL_FILENAME)

      if LOG:
        log_file.write("%d, %d, %d, %.6f\n" % (episode, self.dqn.iteration, train_score, episode_time))
        log_file.flush()



if __name__ == "__main__":
  environment = getEnvironment()
  dqn = DoubleDQN(environment, environment.getActionSet())
  training = DQNTraining(environment, dqn)
  training.train()
