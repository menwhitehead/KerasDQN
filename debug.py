
from keras.models import model_from_json
from keras.optimizers import SGD
import numpy as np

net = model_from_json(open("models/tetrisNet.model").read())
net.load_weights("models/tetrisNet.weights")
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
net.compile(loss='mse', optimizer=sgd)

filters = np.zeros((1, 7))
filters[0][4] = 1.0
filters[0][2] = 1.0

inputs = np.random.rand(1, 1, 16, 8)

out = net.predict([filters, inputs])
print out