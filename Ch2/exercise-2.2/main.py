"""
Exercise 2.3
"""

import numpy as np
import matplotlib.pyplot as plt

num_arms = 10
num_steps = 1000
num_average = 2000
epsilon = 0.1
step_size = 0.5 #set 0 to take averaging step (= 1/N(a))
step_decay_rate = 1e-6
plot_name = 'pick_optimal_action_graph'+ ('_with_fixed_step_size' if step_size!=0 else '')  +('_with_decay' if step_decay_rate!=0 else '') 

low_walk = -10
high_walk = 10

reward_var = 10

q_init = np.tile(np.random.normal(loc=30.0, scale=100.0),num_arms) #initial value
random_walk_steps = np.random.uniform(low_walk,high_walk,(num_steps,num_arms))
#all iterations we average over take the same random walk of the reward means
num_times_taken_optimal = np.zeros(num_steps)
eps_greedy_probab = np.random.uniform(size= (num_average, num_steps))
logical_array = eps_greedy_probab < epsilon

for iteration_num in xrange(num_average):
	Q = np.zeros((num_arms,1))
	N = np.zeros((num_arms,1))
	q = q_init.copy()
	initialise_step  = 0
	for i in xrange(num_steps):
		if initialise_step < num_arms: #initialise the values of Q and N first
			take_action = initialise_step
			reward = np.random.normal(q[take_action],reward_var)
			Q[take_action] += reward
			N[take_action] += 1
			initialise_step += 1
			continue

		q += random_walk_steps[i]#Now start the random walk 

		real_optimal_action = np.argmax(q)	
		est_optimal_action = np.argmax(Q)

		if logical_array[iteration_num,i]: #do epsilon greedy action selection
			take_action =int(np.random.uniform()*(num_arms-1))
		else:
			take_action = est_optimal_action
		reward = np.random.normal(q[take_action],reward_var)
		N[take_action] += 1
		if(step_size==0): #if step_size 0, take averaging step
			step_size = 1.0/N[take_action]
		step_size *= 1-step_decay_rate
		Q[take_action] += step_size*(reward - Q[take_action]) 
		num_times_taken_optimal[i] += (take_action==real_optimal_action)


percent_optimal = num_times_taken_optimal/num_average
if 'percent_optimal_decay' in globals(): 
	print 'with old decay parameter is better ' + str(np.mean(percent_optimal_decay[num_steps/10:] > percent_optimal[num_steps/10:] )*100) + ' percent of the time'
if step_decay_rate!=0:
	percent_optimal_decay = percent_optimal.copy()
plt.plot(percent_optimal)
plt.ylabel('percentage of time taken real optimal action')
plt.xlabel('steps')
plt.savefig(plot_name+'.png')
plt.show()
