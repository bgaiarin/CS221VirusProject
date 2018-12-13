import tensorflow as tf
import numpy as np
import sonnet as snt 

# pass around config object that everything can see? object(self, config)...

class FuncApproximator:

	def sample(self, state): 
		#print ('in sample step in FA with state', state)
		#return [0] * self.num_countries
		# get state to be a numpy array
		npState = np.asarray(state)
		#print (npState)
		#np.reshape(npState, (len(npState), 1))
		#print (npState)
		Us_probs_py, Cs_probs_py = self.sess.run([self.Us_probs, self.Cs_probs], feed_dict={self.state_plc : state})

		# python numpy stuff to sample from the us and cs
		#print('Us', Us_probs_py)
		#print('Cs', Cs_probs_py)

		numUnits = np.random.choice(np.size(Us_probs_py), None, p=np.squeeze(Us_probs_py))
		action = np.random.choice(np.size(Cs_probs_py), numUnits, p=np.squeeze(Cs_probs_py))

		print ('action:', action)

		return action

	# backwards step - do policy gradient update based on how RL controller knows. will have saved all the actions, loop through, call this each time.
	def update(self, state, action, target):
		# transform action into aggregated?
		#print ('in update step in FA with state', state, 'and action', action)
		#Cs, Us = action_to_Cs(action)
		# when return action, also return some hidden info so that the action has the placeholders
		# or iterate over it - trying to go from [0 1 1 2] (aggregate/bucket - what MDP gets) to [1 2 3 3] (input to gather)
		# have a very distinct name for these two!!
		loss_py, _ = self.sess.run([self.loss, self.train], feed_dict={self.state_plc: state, self.action_plc: action, self.target_plc: target})
		#print (loss) # make sure it's, if not going down, at least not going super high
		print('loss in update step:', loss_py)
	# plc = placeholder

	def buildTFgraph(self):
		print ('\n\nin buildTFgraph step in FA\n\n')
		
		MLP_us = snt.nets.MLP(output_sizes=self.Us_output_sizes) # json ish   # units
		MLP_cs = snt.nets.MLP(output_sizes=self.Cs_output_sizes) # countries

		self.state_plc = tf.placeholder(shape=[self.state_dim], dtype=tf.float32, name="state_plc")  #dimension of state

		Us_logits = tf.squeeze(MLP_us(tf.reshape(self.state_plc, [1, self.state_dim])))  # apply MLP

		Cs_logits = tf.squeeze(MLP_cs(tf.reshape(self.state_plc, [1, self.state_dim]))) # squeeze collapses 1 dimensionality

		# logits are 1 x n vectors
		# better to have one MLP and then split at the end (more efficient, statistics etc)

		self.Us_probs = tf.nn.softmax(Us_logits)
		self.Cs_probs = tf.nn.softmax(Cs_logits)

		self.action_plc = tf.placeholder(shape=[None], dtype=tf.int32, name="Us_plc")
		self.target_plc = tf.placeholder(shape=[], dtype=tf.float32, name="Ts_plc") # Ts = future rewards (target)

		Us_lprobs = tf.nn.log_softmax(Us_logits)
		Cs_lprobs = tf.nn.log_softmax(Cs_logits)

		# probability of that number of units
		lprob_of_action = Us_lprobs[tf.size(self.action_plc)] + tf.reduce_sum(tf.gather(Cs_lprobs, self.action_plc))
		self.loss = -(lprob_of_action * self.target_plc)

		self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
		print('\n\nfinished tf graph\n\n')
		# sess.run on train is PG step

	def __init__(self, cfg):
		self.num_countries = cfg["NUM_COUNTRIES"]
		self.num_resources = cfg["NUM_RESOURCES"]

		# output_sizes = tuple(output_sizes) -->  self._num_layers = len(self._output_sizes)
		# --> self._layers = [basic.Linear(self._output_sizes[i]... for each layer
		self.Us_output_sizes = [9, 9, self.num_resources + 1]  # todo add 40 back to first layer
		#self.Us_output_sizes = tf.constant([[40, 20, 1],[1, 0, 0]])	#1 int for # resources	#OR SHOULD IT BE NUM_RESOURCES? 
		self.Cs_output_sizes = [9, 9, self.num_countries]			#OR SOMETHING ELSE? 
		self.state_dim = (self.num_countries * 2) + 1
		self.learning_rate = 0.01
		self.sess = tf.Session() 
		self.buildTFgraph()

		tf.global_variables_initializer().run(session=self.sess)

































######## DUMPSTER ########################################################



'''
RL:
play an episode (keep sampling from mdp and func approx)
save history
for each action, figure out future rewards, sum together
call func approx update 


plot reward per episode - that should go up over a long period of time. ********
loss going down is less likely and less important (still should happen tho)
might be chaotic, but in the right direction
show plot where at every state you take action with highest prob (just take argmax of probability vector)

wants to know how far away we are in expected reward from baselines.
show what this does differently that isn't completely stupid from the baselines.

learning rate is important - start 1e-2 or 1e-3 and look at stability (if too unstable, make smaller and see if it's smoother)
tweak this by waiting 10 min, rather than running overnight.

Still issues that our MDP itself might have shitty rewards system.
keep MLP small enough that it's efficient. reduce hidden layers if needed. better to do more analysis iterations 
than to run experiment longer.
'''



'''
	#h_size = [40, 20]
    def __init__(self, lr, s_size,a_size,h_size):
    	snt.mlp
    	# keras with 2 mlp's, one for number of units and one for per-country allocs. 
    	# pass each through a softmax at the end.

        #These lines establish the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in= tf.placeholder(shape=[None,s_size],dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden,a_size,activation_fn=tf.nn.softmax,biases_initializer=None)
        self.chosen_action = tf.argmax(self.output,1)

        #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.

        # need to manually construct the probability term in log space
        # log of probability of sampling allocation quantity, plus [log (prob of sampling country you allocated for)] for each country you alloc to
        # don't use optimizer - just take the gradients yourself. avoids putting all of this into the tf graph. use tf.gradients
        # hae while loop in tf that will do this for you - will give you log p and action, and then 
        # maximize. 

        self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)
        
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)
        
        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)
        
        self.gradients = tf.gradients(self.loss,tvars)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))
'''
