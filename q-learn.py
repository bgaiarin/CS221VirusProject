from mdp import EpidemicMDP # changed from import mdp
import random 
from collections import defaultdict
import math 
import numpy as np
import matplotlib.pyplot as plt

infections = {'Nigeria' : 1}
resources = 15
# resp_csv = 'data/FR_MAUR_NIG_SA_responseIndicators.csv'
# trans_csv = 'data/FR_MAUR_NIG_SA_transitions.csv'
resp_csv = 'data/country_response_indicators.csv'
#trans_csv = 'data/transitions.csv'
trans_csv = 'data/transitions_7countries.csv'
newmdp = EpidemicMDP(trans_csv, resp_csv, infections, resources) # sorta awk to declare twice but getActions needs instance
NUM_COUNTRIES = newmdp.NUM_COUNTRIES
INDEX_RESOURCE = NUM_COUNTRIES*2
num_simulations = 400
max_iterations = 100
action_without_resources = [[0]*NUM_COUNTRIES]

discount = 1
weights = defaultdict(float)
explorationProb = 0.14
learning_rate = 0.9

#PY PLOT
Xdata = []
Ydata = []
ENDdata = defaultdict(float)



#### Q-LEARNING HELPER FUNCTIONS #################################

# Accepts a state S and discretizes it by rounding response scores to the nearest tenth. 
# Then discretizes it further by grouping binary-response-action integers together for each country,
# and sorting the final list. The idea here is that order should not matter when we're using our 
# states as feature keys in our weights dictionary. 
# ATTENTION: ALL OF THIS IS NOT SOPHISTICATED. 
# Would likely be best to replace this discretization method with a something like 
# multilayer feedforward, or some other NN-backed method. 
def discretizeState(state, action):
    #ROUND TO NEAREST TENTH
    for i in range(NUM_COUNTRIES, INDEX_RESOURCE):
        state[i] = round(state[i], 1)
    #COLLECT ALLOCATION FREQUENCIES
    allocs = [0]*NUM_COUNTRIES
    for a in action: 
        allocs[a] += 1
    #GROUP BINARY-RESPONSE-ACTION TOGETHER FOR EACH STATE, AND SORT
    ds = [None]*NUM_COUNTRIES
    for i in range(NUM_COUNTRIES):
        ds[i] = str(state[i]) + str(state[NUM_COUNTRIES + i]) + str(allocs[i])
    return sorted(ds)

# A helper function. 
# Takes in a state (vector) and returns it as a hashable type (string). 
def makeHashable(state, action):
    state = discretizeState(state, action)
    link = "-"
    return link.join(str(state))

# Return a single-element list containing a binary (indicator) feature
# for the existence of the (state, action) pair.  Provides no generalization.
def featureExtractor(state, action):
    featureKey = makeHashable(state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

#### Q-LEARNING MAIN FUNCTIONS #################################

# Return the Q function associated with the weights and features
def getQ(state, action):
    score = 0
    for f, v in featureExtractor(state, action):
        score += weights[f] * v
    #if (score != 0): print("SCORE: ", score)
    return score

# This algorithm will produce an action given a state.
# Here we use the epsilon-greedy algorithm: with probability
# |explorationProb|, take a random action.
def chooseAction(state, actions):
    if random.random() < explorationProb:
        return random.choice(actions)
    else: 
        max_actions = []
        max_q = float("-inf")
        for a in actions: 
            q = getQ(state, a)
            if (q > max_q):
                max_q = q
                max_actions = [a]
            elif (q == max_q):
                max_actions.append(a)
        return random.choice(max_actions)   #random choice selection for tie-breaking 


# Call this function with (s, a, r, s'), which you should use to update |weights|.
# Note that if s is a terminal state, then s' will be None.  
# Update the weights using getStepSize(). 
# Use getQ() to compute the current estimate of the parameters.
def incorporateFeedback(state, action, reward, newState, actions, num_iterations, newState_is_end):
    Vopt = 0
    if not newState_is_end:      #CHECK: TERMINAL STATE
        for a in actions:
            new_Vopt = getQ(newState, a)
            if (new_Vopt > Vopt): Vopt = new_Vopt

    #update = ((1-getStepSize(num_iterations))*(getQ(state, action))) + ((getStepSize(num_iterations))*(reward + discount*Vopt))
    update = learning_rate*(reward + (discount*Vopt) - getQ(state, action))

    for f, v in featureExtractor(state, action):
        weights[f] += update 


#### SIMULATE (RUN Q-LEARNING) #####################################

def simulateQLearning(trial_num): 
    mdp = EpidemicMDP(trans_csv, resp_csv, infections, resources)
    state = mdp.state   #initial state
    total_rewards = 0
    resources_depleted_delay = 2
    num_iterations = 0
    actions = newmdp.getActions(state)
    for i in range(max_iterations):

        # CASE: VIRUS IS KILLED
        if mdp.isEnd(state): 
            break 

        # CASE: RESOURCES ARE DEPLETED 
        if resources_depleted_delay == 0: break 
        if (state[INDEX_RESOURCE] <= 0):
            resources_depleted_delay -= 1
            actions = action_without_resources      #e.g. [[0,0,0,0]]

        # Choose action based on Q and epsilon-greedy search strategy. 
        best_action = chooseAction(state, actions)
        num_iterations += 1

        # Observe newState and associated reward. 
        newState, reward = mdp.sampleNextState(state, best_action)
        total_rewards += reward

        # Update Q weights 
        actions = newmdp.getActions(newState)
        if (resources_depleted_delay == 2): incorporateFeedback(state, best_action, reward, newState, actions, num_iterations, mdp.isEnd(newState))

        state = newState

    avg_reward = total_rewards/float(num_iterations)
    print("END: ", reward)
    print("AVG: ", avg_reward)

    Xdata.append(trial_num)
    Ydata.append(avg_reward)
    ENDdata[reward] += 1.0


# RUN Q-LEARNING
for i in range(num_simulations):
    #weights = defaultdict(float)   #Reset weights so weights from old simulations don't bleed into new ones. (DO WE NEED TO DO THIS?)
    simulateQLearning(i)
#PLOT 
plt.plot(Xdata, Ydata)
plt.ylabel('Average Reward')
plt.xlabel('Simulation')
plt.title('Average Rewards per Q-Learning Simulation')
plt.show()
#PLOT
keys = ENDdata.keys()
counts = []
for key in keys: 
    counts.append(ENDdata[key])
plt.bar(keys, counts, width=0.8) 
plt.ylabel('Appearances')
plt.xlabel('End Reward')
plt.title('Number of Appearances of End Rewards Across All Q-Learning Simulations')
plt.show()


