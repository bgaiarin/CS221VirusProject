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
NUM_COUNTRIES = 2



def recursiveGetNextStates(state, index, nextStates):
	print '--- entering recursiveGetNextStates at index', index
	if index >= NUM_COUNTRIES: # have made changes to every country
		return
	if state[index] == 0: # infect uninfected state w probability p
		#print 'state at index ', index, 'is', state[index], ', nextStates is', nextStates
		p = 0.1  # dummy value. need info about neighbors - lookup dict?
		newStateFlip = state[:index] + [1] + state[index + 1:]
		nextStates[newStateFlip] = p
		nextStates[state] += (1 - p) # add likelihood of not flipping
	elif state[index] == 1: # cure infected state w probability q
		print 'state at index ', index, 'is', state[index], ', nextStates is', nextStates
		q = state[index + NUM_COUNTRIES]  # get resistance score = prob(cure) = q
		newStateFlip[index] = 0
		nextStates = np.append(nextStates, np.array([(newStateSame, 1 - q), (newStateFlip, q)]))
	else:
		print 'ERROR RIP'
	print 'nextStates at start at index:', index, nextStates
	nextStates = np.append(nextStates, recursiveGetNextStates(newStateSame, index + 1))
	print 'nextStates after appending sames at index:', index, nextStates
	nextStates = np.append(nextStates, recursiveGetNextStates(newStateFlip, index + 1))
	print 'nextStates after appending flips at index:', index, nextStates
	return nextStates

def getStates(state, action):
	nextStates = {state : 0}
	#newState = np.copy(state)
	# alter per-country resistances to reflect new resource allocation based on action. diminishing returns?
	# use y = x / (1+x)

	recursiveGetNextStates(state, 0, nextStates)
	return nextStates

	# all probabilities of states should sum to 1

def getStatesOldNoRecurse(state, action):
	nextStates = [(state, 0)]
	#newState = np.copy(state)
	# alter per-country resistances to reflect new resource allocation based on action. diminishing returns?
	# use y = x / (1+x)

	#return recursiveGetNextStates(state, 0)
	for country in range(NUM_COUNTRIES):
		if state[country] == 0:
			p = 0.1  # dummy value. need info about neighbors - lookup dict?
			newStateFlip = state[:country] + [1] + state[country + 1:]
			nextStates += [(newStateFlip, p)]
			nextStates[0] = (nextStates[0][0], nextStates[0][1] + (1 - p)) # add likelihood of not flipping
		elif state[country] == 1:
			q = state[country + NUM_COUNTRIES]  # get resistance score = prob(cure) = q
			newStateFlip = state[:country] + [0] + state[country + 1:]
			nextStates += [(newStateFlip, q)]
			nextStates[0] = (nextStates[0][0], nextStates[0][1] + (1 - q)) # add likelihood of not flipping
		else:
			print 'ERROR RIP'

	# all probabilities of states should sum to 1
	return nextStates

def recursiveGetNextStates2(state, index):
	print '--- entering recursiveGetNextStates at index', index
	if index >= NUM_COUNTRIES: # have made changes to every country
		return np.array([])  # np.array([np.copy(state)])
	nextStates = np.array([])
	newStateSame = np.copy(state)
	newStateFlip = np.copy(state)
	if state[index] == 0: # infect uninfected state w probability p
		print 'state at index ', index, 'is', state[index], ', nextStates is', nextStates
		p = 0.1  # dummy value. need info about neighbors - lookup dict?
		newStateFlip[index] = 1
		nextStates = np.append(nextStates, np.array([(newStateSame, 1 - p), (newStateFlip, p)]))
	elif state[index] == 1: # cure infected state w probability q
		print 'state at index ', index, 'is', state[index], ', nextStates is', nextStates
		q = state[index + NUM_COUNTRIES]  # get resistance score = prob(cure) = q
		newStateFlip[index] = 0
		nextStates = np.append(nextStates, np.array([(newStateSame, 1 - q), (newStateFlip, q)]))
	else:
		print 'ERROR RIP'
	print 'nextStates at start at index:', index, nextStates
	nextStates = np.append(nextStates, recursiveGetNextStates(newStateSame, index + 1))
	print 'nextStates after appending sames at index:', index, nextStates
	nextStates = np.append(nextStates, recursiveGetNextStates(newStateFlip, index + 1))
	print 'nextStates after appending flips at index:', index, nextStates
	return nextStates




