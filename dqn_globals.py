from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Merge, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, ZeroPadding1D
from keras.layers import merge, Convolution2D, MaxPooling2D, Input
from keras.engine.topology import Layer, InputSpec
import keras.backend as K
import numpy as np
#from environment import Environment
# from minecraft_environment import MinecraftEnvironment
from environments.tetris.tetris_environment import TetrisEnvironment
import time

model_time = time.strftime("%m-%d-%Y_%H-%M")

LEARNING_RATE = 0.01
DECAY = 1e-6
MOMENTUM = 0.9
NETWORK_CLONE_FREQUENCY = 5000  # in iterations
MODEL_FILENAME = "tetrisNet_%s" % model_time
LOAD_EXISTING = False  # Load an existing model for more training
MODEL_SAVE_FREQUENCY = 1000   # Save the model every this number of episodes

FRAME_WIDTH = 22 #20 #84
FRAME_HEIGHT = 10 #10 #84

SEQUENCE_FRAME_COUNT = 1 #4
MINIBATCH_SIZE = 32
GAMMA = 0.95
#NUMBER_ACTIONS = FRAME_HEIGHT - 1 #40 #12
NUMBER_ACTIONS = FRAME_HEIGHT * 4   # 4 is the number of rotations

TRAINING_LOG = "logs/training_log_%s.csv" % model_time
TRAINING_GRAPH = "logs/training_graph_%s.png" % model_time
TESTING_LOG = "logs/testing_log_%s.csv" % model_time


START_EPSILON = 1.0
END_EPSILON = 0.05
EXPLORE = 1000000  # Number of iterations needed for epsilon to reach 0.1

VERBOSE = False
DISPLAY_FREQUENCY = 100  # Display the training progress this often (in terms of episodes)
LOG = True

REPLAY_MEMORY_SIZE = 2000000 #2000000
REPLAY_MEMORY_INIT_SIZE = 1000 # Sequences to add to replay memory before training begins
#SKIP_FRAME = 3 # Number of frames skipped per action
#MODEL = "" # Model file to load
EVALUATE = False # Evaluate the model (no learning)
EVALUATE_EPSILON = 0.00 # epsilon value to use for evaluation
REPEAT_GAMES = 1  # Number of games played in evaluation mode
MAX_EPISODES = 10000000
MAX_FRAMES_PER_EPISODE = 1000


class Identity(Layer):
       
    def __init__(self, **kwargs):
        self.output_dim = NUMBER_ACTIONS
        #self.input_dim = NUMBER_ACTIONS
        super(Identity, self).__init__(**kwargs)

    def build(self, input_shape):
        #input_dim = input_shape[1]
        #initial_weight_value = np.ones((input_dim, self.output_dim))
        #self.Wiggins = K.variable(initial_weight_value)
        #self.trainable_weights = [self.W]
        pass
    
    def call(self, x, mask=None):
        #return K.prod(x, self.Wiggins)
        return K.clip(x, -1000, 1000)  # have to do something so do something stupid
    
    def get_output_shape_for(self, input_shape):
        #assert input_shape and len(input_shape) == 2
        #return (input_shape[0], self.output_dim)
        return (input_shape[0], self.output_dim)

    

def getEnvironment():
    return TetrisEnvironment(FRAME_HEIGHT, FRAME_WIDTH, NUMBER_ACTIONS)

# Minecraft model backup
def createDMModel():
  model = Sequential()
  model.add(Convolution2D(32, 8, 8, subsample=(4,4), activation='relu', input_shape=(4, 84, 84)))
  model.add(Convolution2D(64, 4, 4, subsample=(2,2), activation='relu'))
  model.add(Flatten()) 
  model.add(Dense(512, activation='relu'))
  #model.add(Dropout(0.50))
  model.add(Dense(12, activation='sigmoid'))
  return model


def createModelOrig():
  model_filter = Sequential()
  # model_filter.add(Lambda(lambda x: x, input_shape=(NUMBER_ACTIONS,), output_shape=(NUMBER_ACTIONS,)))
  model_filter.add(Identity(input_shape=(NUMBER_ACTIONS,)))
  
  model = Sequential()
  model.add(ZeroPadding2D((1,1), input_shape=(SEQUENCE_FRAME_COUNT, FRAME_WIDTH, FRAME_HEIGHT)))
  model.add(Convolution2D(32, 3, 3, init="normal", subsample=(1,1), activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(64, 3, 3, init="normal", subsample=(1,1), activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(64, 3, 3, init="normal", subsample=(1,1), activation='relu'))
  model.add(Flatten()) 
  model.add(Dense(512, init="normal", activation='relu'))
  # model.add(Dropout(0.50))
  model.add(Dense(NUMBER_ACTIONS, init="normal", activation='relu'))
  
  merged = Sequential()
  merged.add(Merge([model_filter, model], mode='mul'))

  return merged






# Inception
def createModelInc():
  model_filter = Sequential()
  # model_filter.add(Lambda(lambda x: x, input_shape=(NUMBER_ACTIONS,), output_shape=(NUMBER_ACTIONS,)))
  model_filter.add(Identity(input_shape=(NUMBER_ACTIONS,)))
  
  # INception
  input_img = Input(shape=(SEQUENCE_FRAME_COUNT, FRAME_WIDTH, FRAME_HEIGHT))
  
  tower_1 = Convolution2D(64, 1, 1, border_mode='same', activation='relu')(input_img)
  
  tower_2 = Convolution2D(64, 1, 1, border_mode='same', activation='relu')(input_img)
  tower_2 = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(tower_2)
  
  #tower_3 = Convolution2D(64, 1, 1, border_mode='same', activation='relu')(input_img)
  #tower_3 = Convolution2D(64, 5, 5, border_mode='same', activation='relu')(tower_3)
  
  tower_4 = MaxPooling2D((3, 3), strides=(1, 1), border_mode='same')(input_img)
  tower_4 = Convolution2D(64, 1, 1, border_mode='same', activation='relu')(tower_4)
  
  incept = merge([tower_1, tower_2,  tower_4], mode='concat', concat_axis=1)
  
  inc = Model(input_img, incept)
    
  model = Sequential()
  model.add(inc)
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dense(NUMBER_ACTIONS, activation='relu'))
  
  merged = Sequential()
  merged.add(Merge([model_filter, model], mode='mul'))

  return merged


# Factored Inception
def createModelFactored():
  model_filter = Sequential()
  # model_filter.add(Lambda(lambda x: x, input_shape=(NUMBER_ACTIONS,), output_shape=(NUMBER_ACTIONS,)))
  model_filter.add(Identity(input_shape=(NUMBER_ACTIONS,)))
  
  # Inception
  input_img = Input(shape=(SEQUENCE_FRAME_COUNT, FRAME_WIDTH, FRAME_HEIGHT))
  
  tower_1 = Convolution2D(256, 1, 10, border_mode='same', init="normal", activation='relu')(input_img)
  
  tower_2 = Convolution2D(512, 20, 1, border_mode='same', init="normal", activation='relu')(input_img)
  #tower_2 = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(tower_2)
  
  #tower_3 = Convolution2D(64, 1, 1, border_mode='same', activation='relu')(input_img)
  #tower_3 = Convolution2D(64, 5, 5, border_mode='same', activation='relu')(tower_3)
  
  #tower_4 = MaxPooling2D((3, 3), strides=(1, 1), border_mode='same')(input_img)
  #tower_4 = Convolution2D(64, 1, 1, border_mode='same', activation='relu')(tower_4)
  
  incept = merge([tower_1, tower_2], mode='concat', concat_axis=1)
  
  inc = Model(input_img, incept)
    
  model = Sequential()
  model.add(inc)
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dense(NUMBER_ACTIONS, activation='relu'))
  
  merged = Sequential()
  merged.add(Merge([model_filter, model], mode='mul'))

  return merged



# Two towers
def createModelTwoTowers():
  model_filter = Sequential()
  # model_filter.add(Lambda(lambda x: x, input_shape=(NUMBER_ACTIONS,), output_shape=(NUMBER_ACTIONS,)))
  model_filter.add(Identity(input_shape=(NUMBER_ACTIONS,)))
  
  # Inception
  input_img = Input(shape=(SEQUENCE_FRAME_COUNT, FRAME_WIDTH, FRAME_HEIGHT))
  
  tower1 = Convolution2D(64, 5, 5, border_mode='same', init="normal", activation='relu')(input_img)
  tower2 = Convolution2D(64, 3, 3, border_mode='same', init="normal", activation='relu')(input_img)
  
  tower3 = Convolution2D(64, 3, 3, border_mode='same', init="normal", activation='relu')(tower1)
  tower4 = Convolution2D(64, 3, 3, border_mode='same', init="normal", activation='relu')(tower2)

  #tower_2 = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(tower_2)
  
  #tower_3 = Convolution2D(64, 1, 1, border_mode='same', activation='relu')(input_img)
  #tower_3 = Convolution2D(64, 5, 5, border_mode='same', activation='relu')(tower_3)
  
  #tower_4 = MaxPooling2D((3, 3), strides=(1, 1), border_mode='same')(input_img)
  #tower_4 = Convolution2D(64, 1, 1, border_mode='same', activation='relu')(tower_4)
  
  incept = merge([tower1, tower2, tower3, tower4], mode='concat', concat_axis=1)
  
  inc = Model(input_img, incept)
    
  model = Sequential()
  model.add(inc)
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dense(NUMBER_ACTIONS, activation='relu'))
  
  merged = Sequential()
  merged.add(Merge([model_filter, model], mode='mul'))

  return merged



# Tiny model
def createModelTiny():
  model_filter = Sequential()
  model_filter.add(Identity(input_shape=(NUMBER_ACTIONS,)))
  
  model = Sequential()
  model.add(ZeroPadding2D((1,1), input_shape=(SEQUENCE_FRAME_COUNT, FRAME_WIDTH, FRAME_HEIGHT)))
  model.add(Convolution2D(32, 3, 3, init="normal", subsample=(1,1), activation='relu'))
  model.add(Flatten()) 
  model.add(Dense(64, activation='relu'))
  model.add(Dense(NUMBER_ACTIONS, activation='relu'))
  
  merged = Sequential()
  merged.add(Merge([model_filter, model], mode='mul'))

  return merged





def createModelDeepish():
  model_filter = Sequential()
  # model_filter.add(Lambda(lambda x: x, input_shape=(NUMBER_ACTIONS,), output_shape=(NUMBER_ACTIONS,)))
  model_filter.add(Identity(input_shape=(NUMBER_ACTIONS,)))
  
  model = Sequential()
  model.add(ZeroPadding2D((1,1), input_shape=(SEQUENCE_FRAME_COUNT, FRAME_WIDTH, FRAME_HEIGHT)))
  model.add(Convolution2D(32, 3, 3, init="normal", subsample=(1,1), activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(32, 3, 3, init="normal", subsample=(1,1), activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(64, 3, 3, init="normal", subsample=(1,1), activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(64, 3, 3, init="normal", subsample=(1,1), activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(128, 3, 3, init="normal", subsample=(1,1), activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(128, 3, 3, init="normal", subsample=(1,1), activation='relu'))
  model.add(Flatten()) 
  model.add(Dense(512, init="normal", activation='relu'))
  # model.add(Dropout(0.50))
  model.add(Dense(NUMBER_ACTIONS, init="normal", activation='relu'))
  
  merged = Sequential()
  merged.add(Merge([model_filter, model], mode='mul'))

  return merged




# Worked pretty well
def createModel1Conv():
  model_filter = Sequential()
  # model_filter.add(Lambda(lambda x: x, input_shape=(NUMBER_ACTIONS,), output_shape=(NUMBER_ACTIONS,)))
  model_filter.add(Identity(input_shape=(NUMBER_ACTIONS,)))
  
  model = Sequential()
  model.add(ZeroPadding2D((1,1), input_shape=(SEQUENCE_FRAME_COUNT, FRAME_WIDTH, FRAME_HEIGHT)))
  model.add(Convolution2D(128, 3, 3, init="normal", subsample=(1,1), activation='relu'))
  # model.add(ZeroPadding2D((1,1)))
  # model.add(Convolution2D(64, 3, 3, subsample=(1,1), activation='relu'))
  model.add(Flatten()) 
  model.add(Dense(128, activation='relu'))
  model.add(Dense(NUMBER_ACTIONS, activation='relu'))
  
  merged = Sequential()
  merged.add(Merge([model_filter, model], mode='mul'))

  return merged




# Remove dense, add second conv
def createModel():
  model_filter = Sequential()
  # model_filter.add(Lambda(lambda x: x, input_shape=(NUMBER_ACTIONS,), output_shape=(NUMBER_ACTIONS,)))
  model_filter.add(Identity(input_shape=(NUMBER_ACTIONS,)))
  
  model = Sequential()
  model.add(ZeroPadding2D((1,1), input_shape=(SEQUENCE_FRAME_COUNT, FRAME_WIDTH, FRAME_HEIGHT)))
  model.add(Convolution2D(128, 3, 3, init="normal", subsample=(1,1), activation='relu'))
  
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(64, 3, 3, init="normal", subsample=(1,1), activation='relu'))
  
  #model.add(ZeroPadding2D((1,1)))
  #model.add(Convolution2D(32, 3, 3, init="normal", subsample=(1,1), activation='relu'))
  # model.add(ZeroPadding2D((1,1)))
  # model.add(Convolution2D(64, 3, 3, subsample=(1,1), activation='relu'))
  model.add(Flatten()) 
  model.add(Dense(1024, activation='relu'))
  model.add(Dense(NUMBER_ACTIONS, activation='relu'))
  
  merged = Sequential()
  merged.add(Merge([model_filter, model], mode='mul'))

  return merged




def createModelWide():
  model_filter = Sequential()
  # model_filter.add(Lambda(lambda x: x, input_shape=(NUMBER_ACTIONS,), output_shape=(NUMBER_ACTIONS,)))
  model_filter.add(Identity(input_shape=(NUMBER_ACTIONS,)))
  
  model1 = Sequential()
  model1.add(ZeroPadding2D((1,1), input_shape=(SEQUENCE_FRAME_COUNT, FRAME_WIDTH, FRAME_HEIGHT)))
  model1.add(Convolution2D(32, 7, 7, init="normal", subsample=(1,1), activation='relu'))
  
  model2 = Sequential()
  model2.add(ZeroPadding2D((1,1), input_shape=(SEQUENCE_FRAME_COUNT, FRAME_WIDTH, FRAME_HEIGHT)))
  model2.add(Convolution2D(32, 5, 5, init="normal", subsample=(1,1), activation='relu'))
  
  model3 = Sequential()
  model3.add(ZeroPadding2D((1,1), input_shape=(SEQUENCE_FRAME_COUNT, FRAME_WIDTH, FRAME_HEIGHT)))
  model3.add(Convolution2D(32, 3, 3, init="normal", subsample=(1,1), activation='relu'))

  model.add(Flatten()) 
  model.add(Dense(512, init="normal", activation='relu'))
  # model.add(Dropout(0.50))
  model.add(Dense(NUMBER_ACTIONS, init="normal", activation='relu'))
  
  merged = Sequential()
  merged.add(Merge([model_filter, model], mode='mul'))

  return merged




