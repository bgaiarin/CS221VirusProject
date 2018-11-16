
# ALL CODE BELOW IS FROM BEN GAIARIN'S CS221 PSET (BLACKJACK)

import math, random
from collections import defaultdict

############################################################
# Problem 4a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
       
        #FIGURE OUT Vopt: 
        Vopt = 0
        if (newState != None):      #CHECK: TERMINAL STATE
            for a in self.actions(newState):
                new_Vopt = self.getQ(newState, a)
                if (new_Vopt > Vopt): Vopt = new_Vopt

        update = ((1-self.getStepSize())*(self.getQ(state, action))) + ((self.getStepSize())*(reward + self.discount*Vopt))

        for f, v in self.featureExtractor(state, action):
            self.weights[f] += update 

        # END_YOUR_CODE

# Return a single-element list containing a binary (indicator) feature
# for the existence of the (state, action) pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

############################################################
# Problem 4b: convergence of Q-learning
# Small test case
smallMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# Large test case
largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)

def simulate_QL_over_MDP(mdp, featureExtractor):
    # NOTE: adding more code to this function is totally optional, but it will probably be useful
    # to you as you work to answer question 4b (a written question on this assignment).  We suggest
    # that you add a few lines of code here to run value iteration, simulate Q-learning on the MDP,
    # and then print some stats comparing the policies learned by these two approaches.
    # BEGIN_YOUR_CODE

    tempMdp = mdp

    vi = ValueIteration()
    vi.solve(tempMdp)

    mdp.computeStates()
    rl = QLearningAlgorithm(mdp.actions, mdp.discount(), identityFeatureExtractor, 0.2)
    util.simulate(mdp, rl, 30000)
    rl.explorationProb = 0

    count = 0.0
    diff = 0
    for k, v in vi.pi.iteritems():
        count += 1.0
        a = rl.getAction(k)
        if(a != v): 
            diff += 1 
    # print('# of Differences: ', diff)
    # print('Percent: ', (diff/count))

    # END_YOUR_CODE


############################################################
# Problem 4c: features for Q-learning.

def minimizeCounts(counts):
    arr = list(counts)
    for i, c in enumerate(arr):
        if (arr[i] != 0): arr[i] = 1
    return (tuple(arr))

# You should return a list of (feature key, feature value) pairs.
# (See identityFeatureExtractor() above for a simple example.)
# Include the following features in the list you return:
# -- Indicator for the action and the current total (1 feature).
# -- Indicator for the action and the presence/absence of each face value in the deck.
#       Example: if the deck is (3, 4, 0, 2), then your indicator on the presence of each card is (1, 1, 0, 1)
#       Note: only add this feature if the deck is not None.
# -- Indicators for the action and the number of cards remaining with each face value (len(counts) features).
#       Note: only add these features if the deck is not None.
def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state

    # BEGIN_YOUR_CODE (our solution is 8 lines of code, but don't worry if you deviate from this)
    arr = []
    arr.append(((total, action), 1))
    if (counts != None):
        indicatorcounts = minimizeCounts(counts)
        arr.append(((indicatorcounts, action), 1)) 
        for i, count in enumerate(counts): 
            arr.append(((i, count, action), 1))

    return arr

    # END_YOUR_CODE