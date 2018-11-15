import math, random
from collections import defaultdict
import csv 
import numpy as np
import itertools

responses_csv = "FR_MAUR_NIG_SA_responseIndicators.csv"
transitions_csv = "FR_MAUR_NIG_SA_transitions.csv"
NUM_COUNTRIES = 4
INDEX_RESOURCE = NUM_COUNTRIES*2
LEFTOVER_RESOURCES_WEIGHT = 0.6

# def normalizeResponseIndicators(num):
# 	return (num/(1+num))

# If NUM_COUNTRIES = 4, creates an array of indices: [0, 1, 2, 3]
def getIndexArray(): 
	arr = []
	for i in range(NUM_COUNTRIES):
		arr.append(i)
	return arr 

# Get all possible combinations of actions for a given # of resources 
def iterateGetActions(index_array, resources):
	combos = itertools.combinations_with_replacement(index_array, resources)
	arr = []
	for c in combos: 
		actions = [0]*NUM_COUNTRIES
		for index in c: 
			actions[index] += 1
		arr.append(actions)
	return arr

def getActions(state):
	num_resources = state[INDEX_RESOURCE]
	all_actions = []
	index_array = getIndexArray()
	for i in range(1, num_resources+1):
		actions = iterateGetActions(index_array, i)
		for action in actions: 
			all_actions.append(action)
	return all_actions

def getStates(state, action):

	# TODO alter per-country resistances to reflect new resource allocation based on action. diminishing returns?
	# use y = x / (1+x) along w action

	nextStates = {tuple(state) : state}
	nextProbs = {tuple(state) : 0}

	for index in range(NUM_COUNTRIES):
		for key, newState in nextStates.items():
			if newState[index] == 0:
				p = 0.1           # TODO dummy value. need info about neighbors - lookup dict?
				newStateFlip = newState[:index] + [1] + newState[index + 1:]
				nextStates[tuple(newStateFlip)] = newStateFlip
				nextProbs[tuple(newStateFlip)] = p
				nextProbs[tuple(newState)] += (1 - p) # add likelihood of not flipping
			elif newState[index] == 1:
				q = newState[index + NUM_COUNTRIES]  # get resistance score = prob(cure) = q
													 # TODO make this more sophisticated?
				newStateFlip = newState[:index] + [0] + newState[index + 1:]
				nextStates[tuple(newStateFlip)] = newStateFlip
				nextProbs[tuple(newStateFlip)] = q
				nextProbs[tuple(newState)] += (1 - q) # add likelihood of not flipping
			else:
				print 'INFECTION FLAG NON-BINARY VALUE ERROR FOR COUNTRY AT INDEX', index

	# all probabilities of states should sum to 1, plus need to zip dicts together
	probSum = 0.0
	for key, prob in nextProbs.items():
		probSum += prob
	nextStatesList = []
	for key, state in nextStates.items():
		nextStatesList += [(state, nextProbs[key] / probSum)]
	return nextStatesList

# Takes a number and makes it fall between 1 and 0. 
def normalize(num):
 	return (num/(1.0+num))

# Return reward for a given state, where reward = (# uninfected countries + weight*leftover_resources)
def getReward(state): 
	num_zeros = 0
	leftover = normalize(state[INDEX_RESOURCE])*LEFTOVER_RESOURCES_WEIGHT
	for i in range(0, NUM_COUNTRIES): 
		if state[i] == 0: num_zeros += 1
	return num_zeros + leftover

# TODO some require numpy and some require not-numpy oops
state = np.array([0,1,0,1,0,0,0,0,3])
print getActions(state)
print(getReward(state))
stateNotNP = [0,1,0,1,0,0,0,0,3]
print getStates(stateNotNP, [0,0,0,0])

