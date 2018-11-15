import math, random
from collections import defaultdict
import csv 
import numpy as np
import itertools

responses_csv = 'data/FR_MAUR_NIG_SA_responseIndicators.csv'
transitions_csv = 'data/FR_MAUR_NIG_SA_transitions.csv'
NUM_COUNTRIES = 4
INDEX_RESOURCE = NUM_COUNTRIES*2
LEFTOVER_RESOURCES_WEIGHT = 0.6
INFECTION_COEFFICIENT = 3.0

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

# gets the probability a country is infected as a function of number of seats coming in from infected neighbors
def getInfectionProb(index, state, countries, neighbors):
	country = countries[index]
	infectedSeats = 0
	for neighbor in neighbors[country]:
		if state[countries.index(neighbor[0])] == 1: # if neighbor is infected
			infectedSeats += neighbor[1]
	return infectedSeats * INFECTION_COEFFICIENT / countries[-1]

def sampleNextStateReward(state, action, countries, neighbors):

	# TODO alter per-country resistances to reflect new resource allocation based on action.
	# diminishing returns?
	# use y = x / (1+x) along w action


	newState = state[:]

	for index in range(NUM_COUNTRIES):
		if newState[index] == 0:
			p = getInfectionProb(index, state, countries, neighbors)    # p(infected from neighbors)
			if random.uniform(0,1) < p:
				newState = newState[:index] + [1] + newState[index + 1:]
		elif newState[index] == 1:
			q = newState[index + NUM_COUNTRIES]  # get resistance score = prob(cure) = q
												 # TODO make this more sophisticated?
			print 'print q:', q
			if random.uniform(0,1) < q:
				print 'cured'
				newState = newState[:index] + [0] + newState[index + 1:]
		else:
			print 'INFECTION FLAG NON-BINARY VALUE ERROR FOR COUNTRY AT INDEX', index

	reward = getReward(newState)
	return newState, reward

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

state = [0,1,0,1,0.1,0.3,0.3,0.7,3]
print getActions(state)
print(getReward(state))
print sampleNextStateReward(state, [0,0,0,0], countries, neighbors)

