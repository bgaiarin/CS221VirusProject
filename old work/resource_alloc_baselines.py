import baseline as mdp

state = [0,1,0,1,0.1,0.3,0.3,0.7,3]
print mdp.sampleNextStateReward(state, [0,1,1,0], countries, neighbors)

### UNIFORM RESOURCE ALLOCATION: EVERYTHING AT T=1, EQUAL NUMBERS TO EACH STATE

### EQUAL RESOURCE ALLOCATION: EVERYTHING AT T=1, EQUAL NUMBERS TO EACH INFECTED STATE

### SINGLE RESOURCE ALLOCATION: ONE UNIT PER INFECTED STATE PER TIME SLICE

### RANDOM RESOURCE ALLOCATION: EVERYTHING AT T=1, RANDOM NUMBERS TO EACH STATE
# for each unit, randomly pick index to give it to

### RANDOM RESOURCE ALLOCATION: ONE UNIT PER TIME SLICE
# randomly pick index to give it to

