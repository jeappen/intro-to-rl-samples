import numpy as np
import matplotlib.pyplot as plt

num_arms = 10
num_steps = 1000
num_average = 2000
epsilon = 0.1
step_size = 0.1

action_variance = 10

q = np.random.normal(loc=30.0, scale=100.0, size=num_arms)
real_optimal_action = np.argmax(q)	

num_times_taken_optimal = np.zeros(num_steps)
eps_greedy_probab = np.random.uniform(size= (num_average, num_steps))
logical_array = eps_greedy_probab < epsilon

for iteration_num in xrange(num_average):
	Q = np.zeros((num_arms,1))
	N = np.zeros((num_arms,1))
	for i in xrange(num_steps):
		optimal_action = np.argmax(Q)
		if logical_array[iteration_num,i]:
			take_action =int(np.random.uniform()*(num_arms-1))
		else:
			take_action = optimal_action
		reward = np.random.normal(q[take_action],action_variance)
		N[take_action] += 1
		step_size = 1.0/N[take_action]
		Q[take_action] += step_size*(reward - Q[take_action]) 
		num_times_taken_optimal[i] += (take_action==real_optimal_action)



plt.plot(num_times_taken_optimal/num_average)
plt.ylabel('some numbers')
plt.savefig('pick_optimal_action_graph.png')
plt.show()
