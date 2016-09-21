from keras.models import model_from_json
from keras.optimizers import SGD, Adadelta, Adam
import numpy as np
from dqn_globals import *
from replay_memory import *
import random

class DoubleDQN:
    
    def __init__(self, environment, legal_actions):
        self.environment = environment
        self.legal_actions = legal_actions
        self.replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE)
        self.iteration = 0
        self.clone_frequency = NETWORK_CLONE_FREQUENCY
        self.netA = createModel() #createDMModel()
        self.netB = createModel() #createDMModel()
        # self.optimize = SGD(lr=LEARNING_RATE, decay=DECAY, momentum=MOMENTUM, nesterov=True)
        #self.optimize = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)
        self.optimize = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        #self.target_net = createModel()
        #self.cloneTrainingNetToTargetNet()
        
    def printModel(self):
        for layer in self.net.layers:
            conf = layer.get_config()
            weights = layer.get_weights()
            print self.net.get_weights()
        
    def compileModel(self):
        self.netA.compile(loss='mse', optimizer=self.optimize)
        self.netB.compile(loss='mse', optimizer=self.optimize)

    def loadModel(self, model_filenameA, model_filenameB):
        self.netA = model_from_json(open("models/" + model_filenameA + ".model").read(), custom_objects={"Identity": Identity})
        self.netA.load_weights("models/" + model_filenameA + ".weights")
        self.netB = model_from_json(open("models/" + model_filenameB + ".model").read(), custom_objects={"Identity": Identity})
        self.netB.load_weights("models/" + model_filenameB + ".weights")

    def saveModel(self, model_filenameA, model_filenameB):
        #print "Saving MODEL %s..." % model_filename
        json_string = self.netA.to_json()
        open("models/" + model_filenameA + '.model', 'w').write(json_string)
        self.netA.save_weights("models/" + model_filenameA + '.weights', overwrite=True)
    
        json_string = self.netB.to_json()
        open("models/" + model_filenameB + '.model', 'w').write(json_string)
        self.netB.save_weights("models/" + model_filenameB + '.weights', overwrite=True)

    def convertFramesToVector(self, frames):
        vec = np.stack(frames)
        return vec

    def selectAction(self, frames, epsilon):
        if random.random() < epsilon:
            return random.choice(self.environment.getActionSet())
        else:
            #return self.selectGreedyAction(self.net, frames)
            return self.selectDoubleGreedyActionSoft(self.netA, self.netB, frames)
            
    def generatePredictions(self, net, frames):
        vec = self.convertFramesToVector(frames)
        print vec.shape
        vec = np.reshape(vec, (1, SEQUENCE_FRAME_COUNT, FRAME_WIDTH, FRAME_HEIGHT))
        fake_filter = np.ones((1, NUMBER_ACTIONS))
        outputs = net.predict([fake_filter, vec])
        return outputs
    
    def generateBatchPredictions(self, net, frames):
        fake_filter = np.ones((len(frames), NUMBER_ACTIONS))
        outputs = net.predict([fake_filter, frames])
        return outputs
            
    # Always pick the absolute best action
    def selectGreedyActionHard(self, net, frames):
        outputs = list(self.generatePredictions(net, frames)[0])
        return outputs.index(max(outputs))
    
    # Probabilistically choose an action with both networks
    def selectDoubleGreedyActionSoft(self, netA, netB, frames):
        outputsA = self.generateBatchPredictions(netA, frames)[0]
        outputsB = self.generateBatchPredictions(netB, frames)[0]
        #total_output = sum(outputs)
        #outputs /= total_output
        outputs = np.mean((outputsA, outputsB), axis=0)

        # Spin the roulette wheel
        r = random.random() + 0.00000001
        index = 0
        some = 0
        while some < r and index < len(outputs):
            some += outputs[index]
            index += 1
        
        return index-1

    # Probabilistically choose an action with one network
    def selectGreedyActionSoft(self, netA, frames):
        outputs = self.generateBatchPredictions(netA, frames)
        print outputs
        outputs = np.mean(outputs, axis=0)
        print outputs

        # Spin the roulette wheel
        r = random.random() + 0.00000001
        index = 0
        some = 0
        while some < r and index < len(outputs):
            some += outputs[index]
            index += 1
        
        return index-1


    def update(self):
        self.iteration += 1
        past_frames, actions, rewards, next_frames = self.replay_memory.pickMinibatch()

        targets = np.zeros((MINIBATCH_SIZE, NUMBER_ACTIONS))
        real_filter = np.zeros((MINIBATCH_SIZE, NUMBER_ACTIONS))

        for i in range(len(actions)):
            real_filter[i][int(actions[i])] = 1.0

            if random.random() < 0.0:
                if next_frames[i][0][0][0] == -1:
                    targets[i, int(actions[i])] = rewards[i]
                else:
                    q_valuesApast = self.generateBatchPredictions(self.netA, past_frames)
                    q_valuesBnext = self.generateBatchPredictions(self.netB, next_frames)

                    astar = self.selectGreedyActionSoft(self.netA, next_frames)
                    qa = q_valuesApast[int(actions[i])]
                    qb = q_valuesBnext[int(astar[i])]
                    targets[i, int(actions[i])] = qa + (rewards[i] + GAMMA * qb[i] - qa[i])
                self.netA.fit([real_filter, past_frames], targets, batch_size=MINIBATCH_SIZE, nb_epoch=1, verbose=0)
            
            else:
                if next_frames[i][0][0][0] == -1:
                    targets[i, int(actions[i])] = rewards[i]
                else:
                    q_valuesAnext = self.generateBatchPredictions(self.netA, next_frames)
                    q_valuesBpast = self.generateBatchPredictions(self.netB, past_frames)

                    bstar = self.selectGreedyActionSoft(self.netB, next_frames)
                    qa = q_valuesAnext[int(bstar[i])]
                    qb = q_valuesBnpast[int(actions[i])]
                    targets[i, int(actions[i])] = qb + (rewards[i] + GAMMA * qa[i] - qb[i])
                self.netA.fit([real_filter, past_frames], targets, batch_size=MINIBATCH_SIZE, nb_epoch=1, verbose=0)
        









