import matplotlib.pyplot as plt
import os
import csv
from pathlib import Path
import numpy as np


def main():
    for item in ['bl-env22/']:#, 'bl-env2/', 'vis_obs-env2/']:
        results = []
        dir_name = 'rl-mapless-navigation/figures_ppo/' + item
        file_name = dir_name + 'avg_reward_his.csv'
        a=1
        s=0
        with open(file_name, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                s=s+float(row[0])
                if a%3==0 and a!=0:
                    #print ("yes")
                    results.append(float(row[0]))
                    s=0
                    
                a+=1

        train_time_step = [i*1. for i in range(15000, 15000*len(results) + 1, 15000)]
        plt.plot(train_time_step, results, label="")

        plt.ylabel("PPO Average reward")
        plt.xlabel('Training time step')
        plt.legend(loc='lower right')
        plt.savefig(dir_name + 'avg_reward_his.png')
        plt.clf()


if __name__=='__main__':
    main()
