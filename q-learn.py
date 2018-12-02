from mdp import EpidemicMDP # changed from import mdp

infections = {'France' : 1}
resources = 5
resp_csv = 'data/country_response_indicators.csv'
trans_csv = 'data/transitions.csv'
newmdp = EpidemicMDP(trans_csv, resp_csv, infections, resources) # sorta awk to declare twice but getActions needs instance
print newmdp.countries
NUM_COUNTRIES = newmdp.NUM_COUNTRIES
INDEX_RESOURCE = NUM_COUNTRIES*2
num_trials = 5
num_iterations = 5
discount = 1


# mdp.sampleNextState(state, action)  => returns: (newState, reward)
# mdp.getReward(state)
# mdp.isEnd(state)
# mdp.getActions(state) => returns: [[], [], etc.] list of all possible actions from state

#### Q-LEARNING #################################

def featureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

# Call this function to get the step size to update the weights.
def getStepSize():
    return 1.0 / math.sqrt(self.num_iterations)

def incorporateFeedback(self, state, action, reward, newState):
    #FIGURE OUT Vopt: 
    Vopt = 0
    if (newState != None):      #CHECK: TERMINAL STATE
        for a in self.actions(newState):
            new_Vopt = self.getQ(newState, a)
            if (new_Vopt > Vopt): Vopt = new_Vopt

    update = ((1-getStepSize())*(self.getQ(state, action))) + ((getStepSize())*(reward + discount*Vopt))

    for f, v in featureExtractor(state, action):
        self.weights[f] += update 


#### SIMULATE #####################################

mdp = EpidemicMDP(trans_csv, resp_csv, infections, resources)
s = mdp.state   #initial state
for i in range(num_iterations):











































# ##### OLD SIMULATE CODE FROM BASELINE.PY ########

# def simulate(getActions, trial_num, resp_csv, trans_csv, infections, resources):
#     mdp = EpidemicMDP(trans_csv, resp_csv, infections, resources)
#     s = mdp.state
#     grand_total_rewards = 0
#     total_reward = 0
#     resources_depleted_delay = 2
#     its = 0
#     for i in range(max_iterations):
        
#         if mdp.isEnd(s): break 
#         if resources_depleted_delay == 0: break 

#         if (s[INDEX_RESOURCE] <= 0):
#             resources_depleted_delay -= 1

#         actions = getActions(s)
#         max_reward = float("-inf")
#         max_state = s
#         for action in actions: 
#             new_s, reward = mdp.sampleNextState(s, action)
#             if (reward > max_reward):
#                 max_reward = reward
#                 max_state = new_s
#         s = max_state
#         total_reward += max_reward
#         its += 1

#     if its == 0:
#         print "TRIAL:", trial_num, '- SIMULATION ITERATIONS EQUALS ZERO'
#     else:
#         avg_reward = total_reward/float(its)
#         print "TRIAL:", trial_num, " ", avg_reward


# ### BEST ACTION using simulate()
# print " "
# print("##### USING MDP.GETACTIONS #####") 
# for i in range(num_trials):
#     simulate(newmdp.getActions, i, resp_csv, trans_csv, infections, resources)



# ####### COPIED FROM BLACKJACK Q-LEARNING ########

# class QLearningAlgorithm(util.RLAlgorithm):
#     def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
#         self.actions = actions
#         self.discount = discount
#         self.featureExtractor = featureExtractor
#         self.explorationProb = explorationProb
#         self.weights = defaultdict(float)
#         self.numIters = 0

#     # Return the Q function associated with the weights and features
#     def getQ(self, state, action):
#         score = 0
#         for f, v in self.featureExtractor(state, action):
#             score += self.weights[f] * v
#         return score

#     # This algorithm will produce an action given a state.
#     # Here we use the epsilon-greedy algorithm: with probability
#     # |explorationProb|, take a random action.
#     def getAction(self, state):
#         self.numIters += 1
#         if random.random() < self.explorationProb:
#             return random.choice(self.actions(state))
#         else:
#             return max((self.getQ(state, action), action) for action in self.actions(state))[1]

#     # Call this function to get the step size to update the weights.
#     def getStepSize(self):
#         return 1.0 / math.sqrt(self.numIters)

#     # We will call this function with (s, a, r, s'), which you should use to update |weights|.
#     # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
#     # You should update the weights using self.getStepSize(); use
#     # self.getQ() to compute the current estimate of the parameters.
#     def incorporateFeedback(self, state, action, reward, newState):
       
#         #FIGURE OUT Vopt: 
#         Vopt = 0
#         if (newState != None):      #CHECK: TERMINAL STATE
#             for a in self.actions(newState):
#                 new_Vopt = self.getQ(newState, a)
#                 if (new_Vopt > Vopt): Vopt = new_Vopt

#         update = ((1-self.getStepSize())*(self.getQ(state, action))) + ((self.getStepSize())*(reward + self.discount*Vopt))

#         for f, v in self.featureExtractor(state, action):
#             self.weights[f] += update 


# # Return a single-element list containing a binary (indicator) feature
# # for the existence of the (state, action) pair.  Provides no generalization.
# def identityFeatureExtractor(state, action):
#     featureKey = (state, action)
#     featureValue = 1
#     return [(featureKey, featureValue)]


# def simulate_QL_over_MDP(mdp):



#     rl = QLearningAlgorithm(mdp.actions, mdp.discount(), identityFeatureExtractor, 0.2)
#     util.simulate(mdp, rl, 30000)
#     rl.explorationProb = 0

#     count = 0.0
#     diff = 0
#     for k, v in vi.pi.iteritems():
#         count += 1.0
#         a = rl.getAction(k)
#         if(a != v): 
#             diff += 1 
#     # print('# of Differences: ', diff)
#     # print('Percent: ', (diff/count))

# mdp = 
# simulate_QL_over_MDP(newmdp)




