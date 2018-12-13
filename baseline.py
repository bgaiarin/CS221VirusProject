from mdp import EpidemicMDP # changed from import mdp
import random 

infections = {'Nigeria' : 1}
resources = 15
# resp_csv = 'data/FR_MAUR_NIG_SA_responseIndicators.csv'
# trans_csv = 'data/FR_MAUR_NIG_SA_transitions.csv'
resp_csv = 'data/country_response_indicators.csv'
#trans_csv = 'data/transitions.csv'
trans_csv = 'data/transitions_9countries.csv'
newmdp = EpidemicMDP(trans_csv, resp_csv, infections, resources) # sorta awk to declare twice but getActions needs instance
print(newmdp.countries)
NUM_COUNTRIES = newmdp.NUM_COUNTRIES
INDEX_RESOURCE = NUM_COUNTRIES*2
num_trials = 8
max_iterations = 100


# Given a number of resources, allocates one to each country. 
# If # resources < # countries, then the first (# countries - # resources)
# countries won't receive any resource.
def getUniformActions(state): 
	resources = state[INDEX_RESOURCE]
	action = []
	countries = NUM_COUNTRIES-1
	for i in range(resources):
		if (countries == -1): break
		action.append(countries)
		countries -= 1
	return [action]

# Allocate all resources to all and only infected countries. 
# Will use all resources in first time step. 
# Allocates equally amongst infected states, so long as 
# resources can be divided equally amongst the infected states.
def getEqualActions(state):
	action_indices = []
	for i in range(0, NUM_COUNTRIES):
		if (state[i] == 1): 
			action_indices.append(i)

	resources = state[INDEX_RESOURCE]
	action = []
	index = len(action_indices)-1
	for i in range(resources):
		if (index == -1): 
			index = len(action_indices)-1		#start over, we run until i == resources
		action.append(action_indices[index])
		index -= 1
	return [action]

# Do not allocate any resources
def getNoActions(state): 
	actions = [[]]
	return actions

# Select random action from all possible actions 
# (all possible allocations of possible resource quantities at time t)
def getRandomActions(state):
	actions = newmdp.getActions(state)
	if (actions == []): return []
	else: 
		c = random.choice(actions)
		return [c]


def simulate(actionCommand, trial_num, resp_csv, trans_csv, infections, resources):
	grand_total_rewards = 0
	mdp = EpidemicMDP(trans_csv, resp_csv, infections, resources)
	s = mdp.state
	total_reward = 0
	resources_depleted_delay = 2
	its = 0
	for i in range(max_iterations):
		
		if mdp.isEnd(s): break 
		if resources_depleted_delay == 0: break 

		if (s[INDEX_RESOURCE] <= 0):
			resources_depleted_delay -= 1

		actions = actionCommand(s)
		max_reward = float("-inf")
		max_state = s
		for action in actions: 
			new_s, reward = mdp.sampleNextState(s, action)
			if (reward > max_reward):
				max_reward = reward
				max_state = new_s
		s = max_state
		total_reward += max_reward
		its += 1

	if its == 0:
		print("TRIAL:", trial_num, '- SIMULATION ITERATIONS EQUALS ZERO')
	else:
		avg_reward = total_reward/float(its)
		#print "TRIAL:", trial_num, " ", avg_reward
		print("End: ", max_reward)
		print("Average: ", avg_reward)


### UNIFORM RESOURCE ALLOCATION: EVERYTHING AT T=1, EQUAL NUMBERS TO EACH STATE
print(" ")
print("##### UNIFORM RESOURCE ALLOCATION #####")
for i in range(num_trials):
	simulate(getUniformActions, i, resp_csv, trans_csv, infections, resources)

### EQUAL RESOURCE ALLOCATION: EVERYTHING AT T=1, EQUAL NUMBERS TO EACH INFECTED STATE
print(" ")
print("##### EQUAL RESOURCE ALLOCATION #####") 
for i in range(num_trials):
	simulate(getEqualActions, i, resp_csv, trans_csv, infections, resources)

### NO RESOURCE ALLOCATION: DON'T DO ANYTHING
print(" ")
print("##### NO RESOURCE ALLOCATION #####") 
for i in range(num_trials):
	simulate(getNoActions, i, resp_csv, trans_csv, infections, resources)

### RANDOM ALLOCATION: RANDOM AMOUNTS ASSIGNED TO RANDOM STATES 
print(" ")
print("##### RANDOM RESOURCE ALLOCATION #####") 
for i in range(num_trials):
	simulate(getRandomActions, i, resp_csv, trans_csv, infections, resources)




















######## DUMPSTER ########################################################


### SINGLE RESOURCE ALLOCATION: ONE UNIT PER INFECTED STATE PER TIME SLICE

### RANDOM RESOURCE ALLOCATION: EVERYTHING AT T=1, RANDOM NUMBERS TO EACH STATE
# for each unit, randomly pick index to give it to

### RANDOM RESOURCE ALLOCATION: ONE UNIT PER TIME SLICE
# randomly pick index to give it to



# def simulate(startstate, actionCommand, trial_num):
# 	grand_total_rewards = 0
	
# 	for i in range(max_iterations):
# 		s = startstate
# 		total_reward = 0
# 		resources_depleted_delay = 5
		
# 		while (mdp.isEnd(s) == False and resources_depleted_delay != 0):		#add convergence? #convergence = 0	#acts as a flag--checks for no improvement in reward
# 			if (s[INDEX_RESOURCE] <= 0):
# 				resources_depleted_delay -= 1
# 			actions = actionCommand(s)
# 			max_reward = 0
# 			max_state = s
			
# 			for action in actions: 
# 				new_s, reward = mdp.sampleNextState(s, action)
# 				if (reward > max_reward):
# 					max_reward = reward
# 					max_state = new_s
			
# 			s = max_state
# 			print(s)
# 			total_reward += max_reward		#should be adding rewards of every state chosen 
		
# 		grand_total_rewards = total_reward
	
# 	avg_total_rewards = grand_total_rewards/max_iterations
# 	print(trial_num, avg_total_rewards)
