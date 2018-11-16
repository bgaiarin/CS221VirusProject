import math, random, csv, itertools
from collections import defaultdict
import numpy as np

responses_csv = 'data/FR_MAUR_NIG_SA_responseIndicators.csv'
transitions_csv = 'data/FR_MAUR_NIG_SA_transitions.csv'
NUM_COUNTRIES = 4

INDEX_RESOURCE = NUM_COUNTRIES*2
LEFTOVER_RESOURCES_WEIGHT = 0.6
INFECTION_COEFFICIENT = 3.0
PREVENTION_COST = 0.8
INFECTION_COST = 0.6 # 0 < x < 1, should be <= PREVENTION_COST
MAX_RESPONSE_SCORE = 0.99

def loadFlights(flightdata):
	countries = []
	neighbors = {}
	totalSeats = 0
	with open(flightdata) as flightFile:
		next(flightFile)
		csvReader = csv.reader(flightFile, delimiter=',')
		for row in csvReader:
			dest = row[1].strip()
			if dest not in neighbors.keys():
				neighbors[dest] = []
			neighbors[dest] += [(row[0].strip(), int(row[2]))]
			totalSeats += int(row[2])
			if dest not in countries:
				countries.append(dest)
	countries.append(totalSeats)
	return countries, neighbors, totalSeats

countries, neighbors, totalSeats = loadFlights(transitions_csv)
print neighbors, totalSeats, countries

def getCountryFromIndex(index, countries):
	return countries[index]

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

# Given a  state, returns all possible actions (resource allocations). 
def getActions(state):
	num_resources = state[INDEX_RESOURCE]
	all_actions = []
	index_array = getIndexArray()
	for i in range(1, num_resources+1):
		actions = iterateGetActions(index_array, i)
		for action in actions: 
			all_actions.append(action)
	return all_actions

# Takes a number and makes it fall between 1 and 0. 
def squash(num):
 	return (num/(1.0+num))

# Accepts a state and an action (ex: [0, 1, 0, 2, 0]), and updates response indicators for countries 
# that have been granted additional resources by the action. Returns a state with the updated response indicators.
# NOTE: There is some funky math here. The equation we use for updating is subject to be changed. 
# NOTE: Why do we have "scalar += 1"? We add 1 to any resource allocation that's > 0 to ensure that 
# 		newState[i] gets increased and not decremented. Again, we can choose not to do this if we change
#		the update equation that we use. 
def updateResistances(state, action): 
	newState = state[:]
	for i in range(NUM_COUNTRIES, INDEX_RESOURCE):
		scalar = action[i-NUM_COUNTRIES]
		if (scalar != 0):
			scalar += 1					
			update = newState[i]*scalar*(squash(scalar))					
			newState[i] = ( update if update < 1.0 else MAX_RESPONSE_SCORE )	#Keep in range (0,1)
	return newState

# gets the probability a country is infected as a function of number of seats coming in from infected neighbors
def getInfectionProb(index, state, countries, neighbors):
	country = countries[index]
	infectedSeats = 0
	for neighbor in neighbors[country]:
		if state[countries.index(neighbor[0])] == 1: # if neighbor is infected
			infectedSeats += neighbor[1]
	return infectedSeats * INFECTION_COEFFICIENT / countries[-1]

# generates a next state and its reward probabilistically based on current state
def sampleNextStateReward(state, action, countries, neighbors):
	newState = state[:]

	# first, reduce the resistance scores of infected countries by a coefficient of INFECTION_COST
	# and uninfected countries by a coefficient of PREVENTION_COST (resources spent in most recent time step)
	for i in range(NUM_COUNTRIES):
		if state[i] == 0:
			newState[NUM_COUNTRIES + i] *= PREVENTION_COST
		elif state[i] == 1:
			newState[NUM_COUNTRIES + i] *= INFECTION_COST
		else:
			print 'INFECTION FLAG NON-BINARY VALUE ERROR FOR COUNTRY AT INDEX', index
	print 'after updating scores:',newState

	# TODO next, alter per-country resistances in newState to reflect new resource allocation based on action.
	# use y = x / (1+x) along w action?
	newState = updateResistances(newState, action)
	print 'after updating resistances to handle actions:', newState

	for index in range(NUM_COUNTRIES):
		if newState[index] == 0:
			p = getInfectionProb(index, state, countries, neighbors)    # p(infected from neighbors)
			if random.uniform(0,1) < p:
				newState = newState[:index] + [1] + newState[index + 1:]
		elif newState[index] == 1:
			q = state[index + NUM_COUNTRIES]  # get resistance score = prob(cure) = q
			if random.uniform(0,1) < q:
				newState = newState[:index] + [0] + newState[index + 1:]
		else:
			print 'INFECTION FLAG NON-BINARY VALUE ERROR FOR COUNTRY AT INDEX', index

	reward = getReward(newState)
	return newState, reward

# Return reward for a given state, where reward = (# uninfected countries + weight*leftover_resources)
def getReward(state): 
	num_zeros = 0
	leftover = squash(state[INDEX_RESOURCE])*LEFTOVER_RESOURCES_WEIGHT
	for i in range(0, NUM_COUNTRIES): 
		if state[i] == 0: num_zeros += 1
	return num_zeros + leftover

state = [0,1,0,1,0.1,0.3,0.3,0.7,3]
print state
print getActions(state)
print(getReward(state))
print sampleNextStateReward(state, [0,1,1,0], countries, neighbors)

