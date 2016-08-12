import numpy as np
import time
import Image

from dqn import DQN
from dqn_globals import *


class DQNEvaluation:
  
  def __init__(self, environment, dqn):
    self.environment = environment
    self.dqn = dqn

    if MODEL_FILENAME == '':
      print "NO MODEL FILENAME GIVEN!"
      sys.exit(1)
    self.dqn.loadModel(MODEL_FILENAME)
    self.dqn.compileModel()
    self.past_frames = np.zeros((SEQUENCE_FRAME_COUNT, FRAME_WIDTH, FRAME_HEIGHT))
    self.frame_index = 0
    self.current_game = 0
    
  def saveScreenshot(self, screenshot, frame_number):
    print "SCREENSHOT:", screenshot
    screenshot = screenshot * 255.
    im = Image.fromarray(screenshot)
    im = im.convert('L')
    im.save("screenshots/screenshot%d.png" % frame_number)
    
  def playOneEpisode(self):
    total_score = 0.0
    episode_frame = 0
    while not self.environment.isEpisodeOver() and episode_frame < MAX_FRAMES_PER_EPISODE:
      episode_frame += 1
      current_frame = self.environment.getScreen()
      self.saveScreenshot(current_frame, episode_frame)
      if self.frame_index < SEQUENCE_FRAME_COUNT:
        self.past_frames[self.frame_index] = current_frame
        self.frame_index += 1
        self.environment.performRandomAction()

      elif self.frame_index >= SEQUENCE_FRAME_COUNT:
        self.past_frames = np.vstack((self.past_frames[1:], np.reshape(current_frame, (1, FRAME_WIDTH, FRAME_HEIGHT))))
       
        action = self.dqn.selectAction(self.past_frames, EVALUATE_EPSILON)
        reward = self.environment.performAction(action)
        total_score += reward;
  
        # Normalize score to -1, 0, 1
        if reward != 0:
          reward /= abs(reward)

          
    self.environment.reset()
    return total_score
  
  def runExperiment(self):
    total_score = 0.0
    for i in range(REPEAT_GAMES):
      self.current_game = i
      score = self.playOneEpisode()
      print "Game %d, score %d" % (i, score)
      total_score += score
    print "TOTAL SCORE: ", total_score



if __name__ == "__main__":
  #environment = MinecraftEnvironment(EVALUATE)
  environment = getEnvironment() #TetrisEnvironment()
  dqn = DQN(environment, environment.getActionSet())
  exp = DQNEvaluation(environment, dqn)
  exp.runExperiment()


