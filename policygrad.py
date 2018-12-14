
#### PACKAGES #################################

from mdp import EpidemicMDP 
from funcapprox import FuncApproximator
import random 
from collections import defaultdict
import math 
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

#### INITIALIZE MDP #################################

infections = {'Nigeria' : 1}
resources = 200
resp_csv = 'data/country_response_indicators.csv' #'data/FR_MAUR_NIG_SA_responseIndicators.csv' #'data/country_response_indicators.csv'
trans_csv = 'data/transitions.csv' #'data/FR_MAUR_NIG_SA_transitions.csv' #'data/transitions.csv' data/transitions_7countries.csv'
mdp = EpidemicMDP(trans_csv, resp_csv, infections, resources)

print ('init mdp. num countries is', mdp.NUM_COUNTRIES)


#### GLOBAL VARS #################################
cfg = {}  # config to pass to func approx
cfg["NUM_COUNTRIES"] = mdp.NUM_COUNTRIES
cfg["INDEX_RESOURCE"] = cfg["NUM_COUNTRIES"] * 2
cfg["NUM_RESOURCES"] = resources
cfg["MAX_REWARD"] = mdp.MAX_REWARD
TOTAL_EPISODES = 150000	#2000? Calculate time per episode, and maximize. One episode should be < 11 seconds.
MAX_ITERATIONS = 20
EXTRA_ITERATIONS = 3 # number of steps to run after it reaches end state

Xdata = []
Rdata = []
Ldata = []
WinCount = 0

#### INITIALIZE FA #################################
fa = FuncApproximator(cfg)

#### RUN #################################

#total outer for loop = number total_episodes
for ep in range(TOTAL_EPISODES):
    state = mdp.state
    actions = []  # list of (state, action, reward) pairs
    itersLeft = EXTRA_ITERATIONS
    reward_total = 0
    its = 0.0
    ep_loss = 0.0

    ## FORWARD PASS: experiment and get reward
    for step in range(MAX_ITERATIONS):
        #print ('in ep', ep, 'taking step', step)
        its += 1.0
        if mdp.isEnd(state):
            # step = MAX_ITERATIONS - itersLeft  # runs a few steps after it reaches end state
            # itersLeft -= 1  # make sure we are able to take actions allocating 0 (FA should do this)
            break
    	
        #sample action using FuncApproximator & save
        action = fa.sample(state)
        #take that action using MDP
        new_state, reward = mdp.sampleNextState(state, action)
        reward_total += reward

        actions.append((state, action, reward)) # does reward have to be cumulative?
    	
        state = new_state

    # reverse actions for easy backward pass
    actions.reverse()
	## BACKWARD PASS: use reward to update actions
    for (state, action, target) in actions:
        #print ('updating')
        #get final reward that we had

        #go over all actions that we took, do policy gradient update on that state and the action 
        ep_loss += fa.update(state, action, target)

    ep_avg_reward = reward_total/its
    ep_avg_loss = ep_loss/its
    print ('episode no.', ep, 'with average reward', ep_avg_reward, 'and average loss', ep_avg_loss)
    
    if (reward == cfg["MAX_REWARD"]): WinCount += 1
    Xdata.append(ep)
    Rdata.append(ep_avg_reward)
    Ldata.append(ep_avg_loss)


# #PRINT NUMBER OF EPISODES WHERE VIRUS IS KILLED
print("NUMBER OF EPISODES WHERE VIRUS IS KILLED: ", WinCount)

# #PLOT
plt.plot(Xdata, Rdata)
plt.ylabel('Average Reward')
plt.xlabel('Simulation')
plt.title('Average Rewards for Policy Gradient')
plt.show()
# #PLOT
plt.plot(Xdata, Ldata)
plt.ylabel('Average Loss')
plt.xlabel('Simulation')
plt.title('Loss for Policy Gradient')
plt.show()




































#### OLD WORK #################################################################
#####################################################################
#####################################################################

#####################################################################

# The following is code for a Vanilla Policy Gradient model 
# Retrieved from: https://github.com/awjuliani/DeepRL-Agents/blob/master/Vanilla-Policy.ipynb
# Original retrieved OS code has been modified to fit our problem. 

# NOTES: 
# 1. Will need to use our MDP instead of "gym" environment. 
# 2. Will need to generalize to more than 2 actions. 

#### DISCOUNTED REWARDS #################################

# def discount_rewards(r):
#     """ take 1D float array of rewards and compute discounted reward """
#     discounted_r = np.zeros_like(r)
#     running_add = 0
#     for t in reversed(xrange(0, r.size)):
#         running_add = running_add * gamma + r[t]
#         discounted_r[t] = running_add
#     return discounted_r

# #### DEFINE AGENT CLASS #################################
# # what is this?
# try:
#     xrange = xrange
# except:
#     xrange = range
# # class agent():
# #     def __init__(self, lr, s_size,a_size,h_size):
# #         #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
# #         self.state_in= tf.placeholder(shape=[None,s_size],dtype=tf.float32)
# #         hidden = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
# #         self.output = slim.fully_connected(hidden,a_size,activation_fn=tf.nn.softmax,biases_initializer=None)
# #         self.chosen_action = tf.argmax(self.output,1)

# #         #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
# #         #to compute the loss, and use it to update the network.
# #         self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
# #         self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)
        
# #         self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
# #         self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

# #         self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)
        
# #         tvars = tf.trainable_variables()
# #         self.gradient_holders = []
# #         for idx,var in enumerate(tvars):
# #             placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
# #             self.gradient_holders.append(placeholder)
        
# #         self.gradients = tf.gradients(self.loss,tvars)
        
# #         optimizer = tf.train.AdamOptimizer(learning_rate=lr)
# #         self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))


# #### TRAINING THE AGENT #################################

# tf.reset_default_graph() #Clear the Tensorflow graph.

# myAgent = FuncApproximator(lr=learning_rate,s_size=4,a_size=2,h_size=8) #Load the agent.

# total_episodes = 5000 #Set total number of episodes to train agent on.
# max_ep = 999
# update_frequency = 5

# init = tf.global_variables_initializer()

# # Launch the tensorflow graph
# with tf.Session() as sess:
#     sess.run(init)
#     i = 0
#     total_reward = []
#     total_lenght = []
        
#     gradBuffer = sess.run(tf.trainable_variables())
#     for ix,grad in enumerate(gradBuffer):
#         gradBuffer[ix] = grad * 0
        
#     while i < total_episodes:
#         s = env.reset()
#         running_reward = 0
#         ep_history = []
#         for j in range(max_ep):
#             #Probabilistically pick an action given our network outputs.
#             a_dist = sess.run(myAgent.output,feed_dict={myAgent.state_in:[s]})
#             a = np.random.choice(a_dist[0],p=a_dist[0])
#             a = np.argmax(a_dist == a)

#             s1,r,d,_ = env.step(a) #Get our reward for taking an action given a bandit.
#             ep_history.append([s,a,r,s1])
#             s = s1
#             running_reward += r
#             if d == True:
#                 #Update the network.
#                 ep_history = np.array(ep_history)
#                 ep_history[:,2] = discount_rewards(ep_history[:,2])
#                 feed_dict={myAgent.reward_holder:ep_history[:,2],
#                         myAgent.action_holder:ep_history[:,1],myAgent.state_in:np.vstack(ep_history[:,0])}
#                 grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
#                 for idx,grad in enumerate(grads):
#                     gradBuffer[idx] += grad

#                 if i % update_frequency == 0 and i != 0:
#                     feed_dict= dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
#                     _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
#                     for ix,grad in enumerate(gradBuffer):
#                         gradBuffer[ix] = grad * 0
                
#                 total_reward.append(running_reward)
#                 total_lenght.append(j)
#                 break

        
#             #Update our running tally of scores.
#         if i % 100 == 0:
#             print(np.mean(total_reward[-100:]))
#         i += 1








