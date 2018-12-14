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

		#print ('action:', action)

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
		return abs(loss_py)
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
		self.Us_output_sizes = [60, 40, 20, 9, self.num_resources + 1]  # todo add 40 back to first layer
		#self.Us_output_sizes = tf.constant([[40, 20, 1],[1, 0, 0]])	#1 int for # resources	#OR SHOULD IT BE NUM_RESOURCES? 
		self.Cs_output_sizes = [60, 40, 20, 9, self.num_countries]			#OR SOMETHING ELSE? 
		self.state_dim = (self.num_countries * 2) + 1
		self.learning_rate = 0.0001
		self.sess = tf.Session() 
		self.buildTFgraph()

		tf.global_variables_initializer().run(session=self.sess)



