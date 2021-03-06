import math, random, csv, itertools
from collections import defaultdict
import numpy as np

class EpidemicMDP:

	#################################################################################
	#								GET ACTIONS 									#
	#################################################################################

	# If r = 4, creates an array of indices: [0, 1, 2, 3]
	def getPartsArray(self, r): 
		arr = []
		for i in range(r+1):
			arr.append(i)
		return arr 

	# Get all possible combinations of actions for a given # of resources 
	def getAllocations(self, resource):
		combos = itertools.combinations_with_replacement(self.getPartsArray(self.NUM_COUNTRIES-1), resource)
		return list(combos)

	# Given a  state, returns all possible actions (resource allocations). 
	def getActions(self, state):
		resources = state[self.INDEX_RESOURCE]
		actions = []
		rlist = self.getPartsArray(resources)
		for r in rlist: 
			allocs = self.getAllocations(r)
			for a in allocs: 
				actions.append(list(a))
		return actions

	#################################################################################
	#							UPDATE RESPONSE INDICATORS 							#
	#################################################################################

	# Accepts a state and an action (ex: [0, 1, 0, 2, 0]), and updates response indicators for countries 
	# that have been granted additional resources by the action. Returns a state with the updated response indicators.
	# NOTE: There is some funky math here. The equation we use for updating is subject to be changed. 
	# NOTE: Why do we have "scalar += 1"? We add 1 to any resource allocation that's > 0 to ensure that 
	# 		newState[i] gets increased and not decremented. Again, we can choose not to do this if we change
	#		the update equation that we use. 
	def updateResistances(self, state, action): 

		newState = state[:]
		for country in action:
			index = country + self.NUM_COUNTRIES
			update = newState[index] * self.RESISTANCE_BOOST
			newState[index] = min(update, self.MAX_RESPONSE_SCORE)  #Keep in range (0,max). caps benefit of reward.

		newState[self.INDEX_RESOURCE] -= len(action) # subtracts units used
		return newState


	#################################################################################
	#							GET NEXT STATE, ACTION & REWARD 					#
	#################################################################################

	# gets the probability a country is infected as a function of number of seats coming in from infected neighbors
	def getInfectionProb(self, index, state):
		country = self.countries[index]
		
		#print '\n\n\ngetting infection prob for country ', country, 'with index', index
		#print self.neighbors
		#print '\nneighbors of the country are', self.neighbors[country]
		
		infectedSeats = 0
		if country not in self.neighbors.keys():
			print ('COUNTRY', country, 'HAS NO INBOUND FLIGHTS IN OUR SYSTEM')
			return 0.0
		for neighbor in self.neighbors[country]:
			if state[self.countries.index(neighbor[0])] == 1: # if neighbor is infected
				infectedSeats += neighbor[1]
		return infectedSeats * self.INFECTION_COEFFICIENT / self.TOTAL_SEATS

	# Checks to see if no country contains virus (we have all zeros [0, 0, 0, 0]). Return True/False and count of 1's.
	def noVirus(self, state):
		num_ones = 0
		for i in range(self.NUM_COUNTRIES):
			if (state[i] != 0): 
				num_ones += 1
		if (num_ones == 0): return (True, num_ones)
		return (False, num_ones)

	# Checks to see if a state is a terminal state (is all [0, 0, 0, 0]). 
	def isEnd(self, state):
		return (self.noVirus(state)[0]) 

	# Return reward for a given state, where reward = (# uninfected countries + weight*leftover_resources)
	# If state, however, is a terminal state, we do one of two things: 
		# If virus is killed, we return a big positive reward. 
		# If virus is not killed but resources have been all used, we return a big negative reward. Ending with uninfected countries is a bonus!
	def getReward(self, state): 

		# Check to see if virus has been terminated and get # of infected and uninfected countries 
		virus_terminated, num_infected_countries = self.noVirus(state)
		num_uninfected_countries = self.NUM_COUNTRIES - num_infected_countries

		# END STATE: No more infected countries 
		if (virus_terminated): 
			return self.MAX_REWARD 	#bonus for saving resources: + (state[self.INDEX_RESOURCE]*self.END_RESOURCES_WEIGHT)

		# END STATE: No more resources
		elif (state[self.INDEX_RESOURCE] <= 0): 
			return -self.MAX_REWARD				#todo: little reward for countries that are still uninfected?
			
		# NOT AN END STATE
		else:
			return -self.MAX_REWARD 
			# unit = self.MAX_REWARD/self.NUM_COUNTRIES
			# num_ones = 0.0
			# num_zeros = 0.0
			# for i in range(0, self.NUM_COUNTRIES): 
			# 	if state[i] == 1: num_ones += unit
			# 	else: num_zeros += unit
			# return num_zeros-num_ones 					#add bonus for leftover resources? 


	# Generates a next state, and its response indicators and reward, probabilistically based on current state. 
	def sampleNextState(self, state, action):
		newState = state[:]

		# Reduce the resistance scores of infected countries by a coefficient of INFECTION_COST
		# and uninfected countries by a coefficient of PREVENTION_COST (resources spent in most recent time step).
		for i in range(self.NUM_COUNTRIES):
			if state[i] == 0:
				newState[self.NUM_COUNTRIES + i] *= self.PREVENTION_COST
			elif state[i] == 1:
				newState[self.NUM_COUNTRIES + i] *= self.INFECTION_COST
			else:
				print ('INFECTION FLAG NON-BINARY VALUE ERROR FOR COUNTRY AT INDEX', index)
		#print 'after updating scores:',newState

		# Alter per-country resistances in newState to reflect new resource allocation based on action.
		newState = self.updateResistances(newState, action)
		#print 'after updating resistances to handle actions:', newState

		# Re-sample a listing of infected and uninfected binary values [0, 1, 0, 0, etc.].
		for index in range(self.NUM_COUNTRIES):
			if newState[index] == 0:
				p = self.getInfectionProb(index, state)#, countries, neighbors)    # p(infected from neighbors)
				if random.uniform(0,1) < p:
					newState = newState[:index] + [1] + newState[index + 1:]
			elif newState[index] == 1:
				q = state[index + self.NUM_COUNTRIES]  # get resistance score = prob(cure) = q
				if random.uniform(0,1) < q:
					newState = newState[:index] + [0] + newState[index + 1:]
			else:
				print ('INFECTION FLAG NON-BINARY VALUE ERROR FOR COUNTRY AT INDEX', index)

		reward = self.getReward(newState)
		return newState, reward

	#################################################################################
	#								INITIALIZE MDP 									#
	#################################################################################

	def loadCountryResponses(self, responseData):
		#print '--- loading country response data ---'
		countryResponses = {}
		with open(responseData) as responseFile:
			next(responseFile)
			csvReader = csv.reader(responseFile, delimiter=',')
			for row in csvReader:
				country = row[0]
				data = row[1]
				if len(data) > 0:
					countryResponses[country] = data
		#print countryResponses
		#print '--- done loading country response data ---'
		return countryResponses


	# loads flight info between countries & populates instance variables
	def loadFlights(self, flightdata):
		#print '--- loading flight data ---'
		countries = []
		neighbors = {}
		totalSeats = 0
		with open(flightdata) as flightFile:
			next(flightFile)
			csvReader = csv.reader(flightFile, delimiter=',')
			for row in csvReader:
				origin = row[0].strip()
				if origin not in countries and origin in self.responseScores.keys():
					countries.append(origin)
					# todo does it matter if country has no flights to/from elsewhere in map?
			flightFile.close()
		with open(flightdata) as flightFile:
			next(flightFile)
			csvReader = csv.reader(flightFile, delimiter=',')
			for row in csvReader:
				origin = row[0].strip()
				dest = row[1].strip()
				#print 'reading in row of flight from', origin, 'to', dest
				if origin in self.responseScores and dest in self.responseScores:
					#print 'origin and dest both have response scores'
					if origin in countries and dest in countries:
						#print 'origin and dest are in country list'
						if dest not in neighbors.keys():
							neighbors[dest] = []
						neighbors[dest] += [(origin, int(row[2]))]
						totalSeats += int(row[2])
		#print '--- done loading flight data ---'
		#print countries, neighbors
		return countries, neighbors, totalSeats

	# builds initial state using country response data, infected countries, and # resources
	def initState(self, initial_infections, initial_resources):
		#print '--- initializing state ---'
		state = [initial_resources]
		for country in self.countries:
			state = [0, 0] + state # 2 slots for each country, keeps resources at end

		for country, score in self.responseScores.items():
			#print country, score
			if country in self.countries:
				state[self.countries.index(country) + len(self.countries)] = float(score) / self.RESPONSE_DENOMINATOR
				#print "added country rank. new state ", state
			else:
				pass#print "COUNTRY NOT FOUND IN COUNTRIES WITH FLIGHTS: ", country

		for country, infection in initial_infections.items():
			if country in self.countries:
				index = self.countries.index(country)
				if index >= self.NUM_COUNTRIES or index < 0:
					print ("ERROR INITIALIZING STATE. COUNTRY NOT FOUND: ", country, index)
				else:
					state[self.countries.index(country)] = infection
			else:
				print ('ERROR: INFECTED COUNTRY', country,'NOT IN LIST OF COUNTRIES')
		#print '--- finished initializing state: ', state
		return state

	# initial_infections is dict from country to 1 or 0 (0 optional). initial_resources is scalar.
	def __init__(self, transitions_csv, responses_csv, initial_infections = {}, initial_resources = 0):
		self.responseScores = self.loadCountryResponses(responses_csv)
		self.countries, self.neighbors, self.TOTAL_SEATS = self.loadFlights(transitions_csv)
		self.NUM_COUNTRIES = len(self.countries)
		self.INDEX_RESOURCE = self.NUM_COUNTRIES * 2
		self.RESPONSE_DENOMINATOR = 110.0 # amount response ranking is divided by during parsing; should be > 100
		self.INFECTION_COEFFICIENT = 10.0
		self.PREVENTION_COST = 0.9
		self.INFECTION_COST = 0.60 # 0 < x < 1, should be <= PREVENTION_COST
		self.MAX_RESPONSE_SCORE = 0.98 # todo change to like .8?
		self.MAX_REWARD = 10.0
		self.RESISTANCE_BOOST = 1.5 # amount by which resistance is increased when 1 resource unit is allocated
		self.state = self.initState(initial_infections, initial_resources)


		
