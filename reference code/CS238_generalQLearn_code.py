
# ALL CODE BELOW IS FROM BEN GAIARIN'S CS238 PSET (PROJECT 2) 

import csv 
import numpy as np
import time

#Q-learning algorithm 
# Initial state = S0
# Initialize Q
# Loop through states: 
#    (Loop through actions for that state?) 
#    Get action for that state 
#    Observe new state St+1 and reward Rt 
#    Q(St, At) <-- Q(St, At) + learningrate(Rt + discount(Q(St+1, At)) - Q(St, At))  

#DISCOUNT 
l_rate = 0.8
start_time = None

def implementQLearning(file, discount, ns, na): 
	q = np.zeros((ns+1, na+1))
	with open(file) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=",")
		next(csv_reader)
		for row in csv_reader:
			s = int(row[0])
			a = int(row[1])
			r = int(row[2])
			sp = int(row[3])
			max_next = max(q[sp][action] for action in range(1, na+1))
			q[s][a] = float(q[s][a] + l_rate*(r + discount*(max_next) - q[s][a]))
	return q

def writePolicy(q, f): 
	file = open(f, "w+")
	for i, state in enumerate(q): 
		if (i == 0): continue 
		max_a = 0
		max_q = float("-inf")
		for action, val in enumerate(state): 
			if (action == 0): continue
			if (val > max_q): 
				max_q = val
				max_a = action
		file.write("%d\n" % max_a)
	file.close()


def main(file): 
	if (file == "small.csv"): 
		discount = 0.95 
		num_states = 100
		num_actions = 4
		write_f = "small.policy"
	elif (file == "medium.csv"): 
		discount = 1
		num_states = 50000
		num_actions = 7
		write_f = "medium.policy"
	else: 
		discount = 0.95 
		num_states = 312020 
		num_actions = 9 
		write_f = "large.policy"

	q = implementQLearning(file, discount, num_states, num_actions)
	writePolicy(q, write_f)

#RUN
start_time = time.time()
main("small.csv")
print("For small.csv  --- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
main("medium.csv")
print("For medium.csv  --- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
main("large.csv")
print("For large.csv  --- %s seconds ---" % (time.time() - start_time))