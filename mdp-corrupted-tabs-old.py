
'''
- One function getActions(state) that takes in a state vector (of the
type [0, 1, 0, 0.6, 0.2, 0.8, 12] we discussed, with data on binary infection,
resistance score, and remaining resources) and returns a list of action vectors
(of the type [1, 2, 0] with number of resources to allocate to each country)
- One function getStates(state, action) that takes in the current state and an
action and returns the different possible next states and each next state's associated
	transition probability
---> randomness is introduced here in 2 ways: 1) a country that is infected becomes
	  un-infected with probability proportional to their resistance score, and
	  2) a country that is not infected becomes infected with probability
	  proportional to the amount of flights coming in from neighboring infected
	  countries
---> side note: a state where every infection flag is 0 is an end state.
- One function getReward(state) that takes in a state and returns a scalar reward value
	for that state
'''
import numpy as np
NUM_COUNTRIES = 3

def recursiveGetNextStates(state, index):
    if index >= NUM_COUNTRIES:  # have made changes to every country
        return np.array([np.copy(state)])
    nextStates = np.array([])
    newStateSame = np.copy(state) # probs don't need this
    newStateFlip = np.copy(state)
    if state[index] == 0: # infect uninfected state w probability p
        p = 0.1  #????    need info about neighbors - lookup dict?
		newStateFlip[index] = 1
		nextStates = np.append(nextStates, [(newStateSame, 1 - p), (newStateFlip, p)])
    elif state[index] == 1: # cure infected state w probability q
        q = state[index + NUM_COUNTRIES]  # get resistance score = prob(cure)
        newStateFlip[index] = 0
        nextStates = np.append(nextStates, [(newStateSame, 1 - q), (newStateFlip, q)])
    else:
        print 'ERROR RIP'
    return nextStates


def getStates(state, action):
 	nextStates = np.array([])
 	newState = np.copy(state)
 	# alter per-country resistances to reflect new resource allocation based on action. diminishing returns?
 	return recursiveGetNextStates(state, 0)
 	# all probabilities of states should sum to 1

state = np.array([0, 1, 0, 0.2, 0.4, 0.6, 3])
print getStates(state, [0, 1, 0])