#!/usr/bin/env python
import matplotlib.pyplot as plt
import csv
import time
import os
import sys
from dqn_globals import *

def getColumn(filename, column):
    results = csv.reader(open(filename), delimiter=",")
    return [result[column] for result in results]

def plot_chart(chart_type, path_to_png, log_file):

    episode = getColumn(log_file,0)
    iteration = getColumn(log_file,1)
    score = getColumn(log_file,2)
    time = getColumn(log_file,3)
    #head, rom = os.path.split(epoch[0]) #first row/column contains the rom file used
    #steps_per_epoch = score[0]; # second column first row is the number of steps in a training epoch
    if len(time) <= 3:  # don't start plotting until there are 2 points to plot
        hours = "0"
    else:
        hours = str(time[-1]);

    #title = "training for %.4f seconds" % sum(time) 
    
    chunk_size = 1000
    scores = map(float, score[2:])
    normalized_scores = []
    normalized_episodes = []
    for i in range(0, len(scores)-chunk_size):
        normalized_scores.append(sum(scores[i:i+chunk_size]) / float(chunk_size))
        normalized_episodes.append(episode[i+1])

    plotables_episodes = []
    plotables_scores = []
    step_size = int(max(1, len(normalized_episodes) / 1000.))
    for i in range(0, len(normalized_episodes), step_size):
        plotables_episodes.append(normalized_episodes[i])
        plotables_scores.append(normalized_scores[i])


    #normalized_epochs = range(len(normalized_scores))
    # normalized_episodes = episode[2:-chunk_size]

    #print len(normalized_scores), len(normalized_episodes)
    
    plt.ion()
    #plt.pause(0.01)
    plt.figure("Average Episode Scores")
    plt.clf()

    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Average Episode Scores")
        
    if len(episode) > 3:
        #plt.plot(epoch[2:], score[2:], linewidth = 2)
        #plt.plot(normalized_episodes, normalized_scores, linewidth = 2)
        plt.plot(plotables_episodes, plotables_scores, linewidth = 2)
        plt.savefig(path_to_png)
        
    plt.show()
    plt.pause(3)



if __name__ == '__main__':
    while True:
        #plot_chart(0, TRAINING_GRAPH, TRAINING_LOG)
        plot_chart(0, sys.argv[1], sys.argv[2])
