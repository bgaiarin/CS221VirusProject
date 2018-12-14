
#### PACKAGES #################################

from mdp import EpidemicMDP 
from funcapprox import FuncApproximator
import random 
from collections import defaultdict
import math 
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

#### INITIALIZE MDP #################################

infexions = {'Nigeria' : 1}
resources = 15
resp_csv = 'data/country_response_indicators.csv' #'data/FR_MAUR_NIG_SA_responseIndicators.csv' #'data/country_response_indicators.csv'
trans_csv = 'data/transitions.csv' #'data/FR_MAUR_NIG_SA_transitions.csv' #'data/transitions.csv' data/transitions_7countries.csv'
init_mdp = EpidemicMDP(trans_csv, resp_csv, infexions, resources)

print ('init mdp. num countries is', init_mdp.NUM_COUNTRIES)


#### GLOBAL VARS #################################
cfg = {}  # config to pass to func approx
cfg["NUM_COUNTRIES"] = init_mdp.NUM_COUNTRIES
cfg["INDEX_RESOURCE"] = cfg["NUM_COUNTRIES"] * 2
cfg["NUM_RESOURCES"] = resources
cfg["MAX_REWARD"] = init_mdp.MAX_REWARD
TOTAL_EPISODES = 150000	#2000? Calculate time per episode, and maximize. One episode should be < 11 seconds.
MAX_ITERATIONS = 20
EXTRA_ITERATIONS = 3 # number of steps to run after it reaches end state

Xdata = []
Rdata = []
Ldata = []
WinCount = 0

#### INITIALIZE FA #################################
fa = FuncApproximator(cfg)

#### RUN #################################

#total outer for loop = number total_episodes
for ep in range(TOTAL_EPISODES):
    
    #Select random initial seed
    # countries = init_mdp.countries
    # arr = []
    # for c in countries:
    #     arr.append({c : 1})
    # infections = random.choice(arr)
                                                # infections = random.sample(arr, random.randint(0,cfg["NUM_COUNTRIES"]-1))
                                                # s = ","
                                                # infections = s.join(infections)
    infections = infexions
    
    mdp = EpidemicMDP(trans_csv, resp_csv, infections, resources)
    state = mdp.state
    actions = []  # list of (state, action, reward) pairs
    itersLeft = EXTRA_ITERATIONS
    reward_total = 0
    its = 0.0
    ep_loss = 0.0

    ## FORWARD PASS: experiment and get reward
    for step in range(MAX_ITERATIONS):
        #print ('in ep', ep, 'taking step', step)
        its += 1.0
        if mdp.isEnd(state):
            # step = MAX_ITERATIONS - itersLeft  # runs a few steps after it reaches end state
            # itersLeft -= 1  # make sure we are able to take actions allocating 0 (FA should do this)
            break
    	
        #sample action using FuncApproximator & save
        action = fa.sample(state)
        #take that action using MDP
        new_state, reward = mdp.sampleNextState(state, action)
        reward_total += reward

        actions.append((state, action, reward)) # does reward have to be cumulative?
    	
        state = new_state

    # reverse actions for easy backward pass
    actions.reverse()
	## BACKWARD PASS: use reward to update actions
    if (ep % 100 == 0):
        print(" ")
        print(" ")
    for (state, action, target) in actions:
        #print ('updating')
        #get final reward that we had

        if (ep % 100 == 0):
            print("EPISODE: ", ep)
            print("STATE: ", state)
            print("ACTION: ", action, " REWARD: ", target)

        #go over all actions that we took, do policy gradient update on that state and the action 
        ep_loss += fa.update(state, action, target)

    ep_avg_reward = reward_total/its
    ep_avg_loss = ep_loss/its
    #print ('episode no.', ep, 'with average reward', ep_avg_reward, 'and average loss', ep_avg_loss)
    
    if (reward == cfg["MAX_REWARD"]): WinCount += 1
    Xdata.append(ep)
    Rdata.append(ep_avg_reward)
    Ldata.append(ep_avg_loss)


# #PRINT NUMBER OF EPISODES WHERE VIRUS IS KILLED
print("NUMBER OF EPISODES WHERE VIRUS IS KILLED: ", WinCount)

# #PLOT
plt.plot(Xdata, Rdata)
plt.ylabel('Average Reward')
plt.xlabel('Simulation')
plt.title('Average Rewards for Policy Gradient')
plt.show()
# #PLOT
plt.plot(Xdata, Ldata)
plt.ylabel('Average Loss')
plt.xlabel('Simulation')
plt.title('Loss for Policy Gradient')
plt.show()









