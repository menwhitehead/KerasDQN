from keras.models import model_from_json
from keras.optimizers import SGD, Adadelta
import numpy as np
from dqn_globals import *
import random
import cv2
import sys

mode_filename = sys.argv[1]

net = createModel() #createDMModel()
optimize = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)
net.compile(loss='mse', optimizer=optimize)

net = model_from_json(open("%s.model" % mode_filename).read(), custom_objects={"Identity": Identity})
net.load_weights("%s.weights" % mode_filename)

weights = net.get_weights()[0]
print weights.shape
for filter_number in range(weights.shape[0]):
  curr_filter = weights[filter_number]
  curr_filter = np.reshape(curr_filter, (1, 10))
  curr_filter *= 255.
  img = cv2.resize(curr_filter, None, fx=40, fy=40, interpolation = cv2.INTER_NEAREST)
  cv2.imwrite("filters/filter%d.png" % filter_number, img)
  
  
        
