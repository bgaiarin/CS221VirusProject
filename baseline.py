import mdp 

NUM_COUNTRIES = 4
INDEX_RESOURCE = NUM_COUNTRIES*2
num_trials = 5
max_iterations = 5

def getDumbActions(state):
	resources = state[INDEX_RESOURCE]
	actions = [[0]*NUM_COUNTRIES]
	actions[0][0] = resources  			#ONLY WORKS IF STATE[0] IS UNINFECTED; IF NOT, CHANGE INDEX HERE.
	return actions 

def getEqualActions(state):
	resources = state[INDEX_RESOURCE]
	num_ones = 0
	action_indices = []
	for i in range(0, NUM_COUNTRIES):
		if (state[i] == 1): 
			num_ones += 1
			action_indices.append(i)
	allocation = float(resources)/num_ones
	actions = [[0]*NUM_COUNTRIES]
	for index in action_indices: 
		actions[0][index] = allocation
	return actions


def getUniformActions(state): 
	resources = state[INDEX_RESOURCE]
	allocation = float(resources)/NUM_COUNTRIES
	actions = [[]]
	for i in range(NUM_COUNTRIES):
		actions[0].append(allocation)
	return actions

def simulate(startstate, actionCommand, trial_num):
	grand_total_rewards = 0
	
	s = startstate
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
			new_s, reward = mdp.sampleNextStateReward(s, action)
			if (reward > max_reward):
				max_reward = reward
				max_state = new_s
		s = max_state
		total_reward += max_reward
		its += 1

	avg_reward = total_reward/its 
	print "TRIAL:", trial_num, " ", avg_reward

### UNIFORM RESOURCE ALLOCATION: EVERYTHING AT T=1, EQUAL NUMBERS TO EACH STATE
print " "
print("##### UNIFORM RESOURCE ALLOCATION #####")
startstate = [0,1,0,0,0.8,0.2,0.88,0.3,5]
for i in range(num_trials):
	simulate(startstate, getUniformActions, i)

### EQUAL RESOURCE ALLOCATION: EVERYTHING AT T=1, EQUAL NUMBERS TO EACH INFECTED STATE
print " "
print("##### EQUAL RESOURCE ALLOCATION #####") 
for i in range(num_trials):
	simulate(startstate, getEqualActions, i)


### DUMB RESOURCE ALLOCATION: EVERYTHING AT T=1, GIVE ALL TO ONE STATE
print " "
print("##### DUMB RESOURCE ALLOCATION #####") 
for i in range(num_trials):
	simulate(startstate, getDumbActions, i)

### BEST ACTION
print " "
print("##### USING MDP.GETACTIONS #####") 
for i in range(num_trials):
	simulate(startstate, mdp.getActions, i)


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
# 				new_s, reward = mdp.sampleNextStateReward(s, action)
# 				if (reward > max_reward):
# 					max_reward = reward
# 					max_state = new_s
			
# 			s = max_state
# 			print(s)
# 			total_reward += max_reward		#should be adding rewards of every state chosen 
		
# 		grand_total_rewards = total_reward
	
# 	avg_total_rewards = grand_total_rewards/max_iterations
# 	print(trial_num, avg_total_rewards)
