from keras.models import model_from_json
from keras.optimizers import SGD, Adadelta
# from keras_functions import *
import numpy as np
from dqn_globals import *
from replay_memory import *
import random

class DQN:
    
    def __init__(self, environment, legal_actions):
        self.environment = environment
        self.legal_actions = legal_actions
        self.replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE)
        self.iteration = 0
        self.clone_frequency = NETWORK_CLONE_FREQUENCY
        self.net = createModel() #createDMModel()
        # self.optimize = SGD(lr=LEARNING_RATE, decay=DECAY, momentum=MOMENTUM, nesterov=True)
        self.optimize = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)

        self.target_net = createModel()
        self.cloneTrainingNetToTargetNet()
        
    def printModel(self):
        for layer in self.net.layers:
            conf = layer.get_config()
            weights = layer.get_weights()
            print self.net.get_weights()
        
    def compileModel(self):
        self.net.compile(loss='mse', optimizer=self.optimize)

    def loadModel(self, model_filename):
        self.net = model_from_json(open("models/" + model_filename + ".model").read(), custom_objects={"Identity": Identity})
        self.net.load_weights("models/" + model_filename + ".weights")
        
    def saveModel(self, model_filename):
        #print "Saving MODEL %s..." % model_filename
        json_string = self.net.to_json()
        open("models/" + model_filename + '.model', 'w').write(json_string)
        self.net.save_weights("models/" + model_filename + '.weights', overwrite=True)
    
    def convertFramesToVector(self, frames):
        #print len(frames)
        vec = np.stack(frames)
        #vec = np.reshape(vec, (1, 4, 84, 84))
        return vec

    def selectAction(self, frames, epsilon):
        if random.random() < epsilon:
            return random.choice(self.environment.getActionSet())
        else:
            #return self.selectGreedyAction(self.net, frames)
            return self.selectGreedyActionSoft(self.net, frames)
            
    def generatePredictions(self, net, frames):
        vec = self.convertFramesToVector(frames)
        vec = np.reshape(vec, (1, SEQUENCE_FRAME_COUNT, FRAME_WIDTH, FRAME_HEIGHT))
        fake_filter = np.ones((1, NUMBER_ACTIONS))
        # fake_filter = np.array([1.0] * NUMBER_ACTIONS)
        outputs = net.predict([fake_filter, vec])
        return outputs
    
    def generateBatchPredictions(self, net, frames):
        #stacked = np.stack(frames)
        #vec = np.reshape(stacked, (MINIBATCH_SIZE, SEQUENCE_FRAME_COUNT, FRAME_SIZE, FRAME_SIZE))
        #print "PAST FRAMES:", past_frames
        # fake_filter = np.array([1.0] * NUMBER_ACTIONS)
        fake_filter = np.ones((len(frames), NUMBER_ACTIONS))
        #fake_filter = np.array([1.0] * NUMBER_ACTIONS)
        # print fake_filter
        outputs = net.predict([fake_filter, frames])
        #print "PREDICTINOS:", outputs
        #max_outputs = outputs.max(1)
        return outputs
            
    # Always pick the absolute best action
    def selectGreedyActionHard(self, net, frames):
        outputs = list(self.generatePredictions(net, frames)[0])
        # print "network outputs:", outputs
        # Choose highest Q value
        # print "Max output:", outputs.index(max(outputs))
        return outputs.index(max(outputs))
    
    # Probabilistically choose an action
    def selectGreedyActionSoft(self, net, frames):
        outputs = self.generatePredictions(net, frames)[0]
        total_output = sum(outputs)
        outputs /= total_output

        # Spin the roulette wheel
        r = random.random() + 0.00000001
        index = 0
        some = 0
        while some < r and index < len(outputs):
            some += outputs[index]
            index += 1
            
        # print "outputs:", outputs
        # print "r:", r
        # print "index:", index
            
        
        return index-1


    def update(self):
        
        self.iteration += 1
        
        if self.iteration % self.clone_frequency == 0:
            self.cloneTrainingNetToTargetNet()
        
        past_frames, actions, rewards, next_frames = self.replay_memory.pickMinibatch()
        # print "REWARDS:", rewards
        # print "ACTIONS:", actions

        q_values = self.generateBatchPredictions(self.target_net, next_frames)
        maxes = q_values.argmax(1)
        # print "Q VALUEDS::", q_values, maxes
        
        targets = np.zeros((MINIBATCH_SIZE, NUMBER_ACTIONS))
        real_filter = np.zeros((MINIBATCH_SIZE, NUMBER_ACTIONS))

        for i in range(len(q_values)):
            real_filter[i][int(actions[i])] = 1.0
            if next_frames[i][0][0][0] == -1:
                targets[i, int(actions[i])] = rewards[i]
            else:
                # targets[i, int(actions[i])] = rewards[i] + GAMMA * maxes[i]
                targets[i, int(actions[i])] = rewards[i] + GAMMA * q_values[i][maxes[i]]

        # print targets[-1]
       
        # print "ALL THAT:"
        # print real_filter
        # print past_frames
        # print targets
       
        self.net.fit([real_filter, past_frames], targets, batch_size=MINIBATCH_SIZE, nb_epoch=1, verbose=0)
        # outputs = self.net.predict([real_filter, past_frames])
        # print "Targets:", outputs


    def cloneTrainingNetToTargetNet(self):
        print "CLONING NET TO TARGET NET"
        self.target_net.set_weights(self.net.get_weights())
        #self.target_net =  self.net #self.net.copyNet()



