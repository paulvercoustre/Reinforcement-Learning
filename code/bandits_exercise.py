#### Bandit exercise ####

# README if libraries need to be installed 
# You need to install numpy and matplotlib using pip (or pip3).
# This can be done with the following commands (in a terminal in a Linux OS):
#   If running Python 2:
#    pip install numpy
#    pip install matplotlib
#   If running Python 3:
#    pip3 install numpy
#    pip3 install cairocffi
#    pip3 install matplotlib
# For this, 'pip' (or pip3) needs to be installed (which is usually already the case). If not, you can do it by installing classically python-dev (for pip) or python3-pip (for pip3), with your usual OS library management tool (yum, aptitude, apt-get, synaptic, ...). If using Python 3, you might need to install libffi-dev as well.
# This version was tested under Python 2.7.6 and under Python 3.4.3.

import numpy as np
import matplotlib.pyplot as plt
import sys

class Banditos:
    def __init__(self, N, k):
        self.cur = 0
        self.q_stars = np.random.randn(N, k)

    def select(self, n):
        self.cur = n

    def act(self, a):
        mean = self.q_stars[self.cur, a]
        reward = mean + np.random.randn()
        return reward

        
class randomAgent:
    def __init__(self, A):
        self.A = A

    def interact(self):
        return np.random.randint(0, self.A)

    def update(self, a, r):
        pass


class epsilonGreedyAgent:
    def __init__(self, A, epsilon):
        self.A = A
        self.epsilon = epsilon
        self.mean = np.zeros(A)
        self.visits = np.zeros(A)
    
    def interact(self):
        if (np.random.uniform() < self.epsilon):
            return np.random.randint(0,self.A)
        else:
            return np.argmax(self.mean)
    
    def update(self, a, r):
        self.mean[a] = (self.mean[a] * self.visits[a]+r)/(self.visits[a]+1)
        self.visits[a] = self.visits[a] + 1

# Running with alpha=0 generates results similar to the randomAgent
# Running with alpha=5 generates results similar to the greedyAgent
class softmaxAgent:
    def __init__(self, A, alpha):
        self.A = A
        self.alpha = alpha
        self.prob = np.zeros(A)
        self.mean = np.zeros(A)
        self.visits = np.zeros(A)

    def interact(self):
        self.prob = np.exp(self.mean * self.alpha)/np.sum(np.exp(self.mean * self.alpha),0)
        # Randomly draws an arm from the empirical probability distribution self.prob
        a = np.random.choice(range(len(self.mean)), p = self.prob) 
        return a
    
    def update(self, a, r):
        self.mean[a] = (self.mean[a] * self.visits[a]+r)/(self.visits[a]+1)
        self.visits[a] = self.visits[a] + 1

# By initiating the arms'mean rewards with too high values, we "force" the agent to try 
# all the arms at least once 
class optimisticEpsilonGreedyAgent:
    def __init__(self, A, epsilon):
        self.A = A
        self.epsilon = epsilon
        # Initiating self.mean with optimisticaly high values
        self.mean = np.ones(A) + 3 
        self.visits = np.zeros(A)
    
    def interact(self):
        if (np.random.uniform() < self.epsilon):
            return np.random.randint(0,self.A)
        else:
            return np.argmax(self.mean)
        return
    
    def update(self, a, r):
        self.mean[a] = (self.mean[a] * self.visits[a]+r)/(self.visits[a]+1)
        self.visits[a] += 1
   
     
class ucbAgent:
    def __init__(self, A):
        self.A = A
        self.mean = np.zeros(A)
        self.visits = np.zeros(A)
        self.upperbound = np.zeros(A)
    
    def interact(self):
        # Computing the upperbound of all arms
        self.upperbound = self.mean + np.sqrt((2 * np.log(np.sum(self.visits)))/self.visits)
        return np.argmax(self.upperbound)
    
    def update(self, a, r):
        self.mean[a] = (self.mean[a] * self.visits[a]+r)/(self.visits[a]+1)
        self.visits[a] += 1
        

# Teacher solution to epsGreedyAgent
class epsGreedyAgent:
    def __init__(self, A, epsilon):
        self.epsilon = epsilon
        self.A = A
        self.cumvalue = np.zeros(A)
        self.numpick = np.zeros(A)

    def interact(self):
        rand = np.random.uniform()
        if rand < self.epsilon:
            a = np.random.randint(0, self.A)
        else:
            value = [c / (n > 0 and n or 1) for c, n
                     in zip(self.cumvalue, self.numpick)]
            a = np.argmax(value)
        return a

    def update(self, a, r):
        self.numpick[a] += 1
        self.cumvalue[a] += r

"""
Create your own agent classes implementing
first the epsilon greedy agent, and then the
Softmax agent, optimisticEpsGreedyAgent
and UCBAgent.
To make your classes compatible with the
tester, they must exhibit a constructor of
the form
def __init__(self, A, ...):

with ... being other parameters of your choice,
a function
def interact(self):

that returns an action given the current state
of the bandit, and a function
def update(self, a, r):

that takes the action that was performed, the
reward that was obtained, and updates the state
of the bandit. The epsGreedyAgent is here to
help you get an idea on how to implement these
methods.

Once your implementation of an agent is complete,
you can test it by replacing randomAgent in
the AgentTester parameters in the main script below
by your own class, and give a table containing the
parameters you want to use as a dictionnary (e.g.
{'epsilon': 0.1}) as an argument.

The AgentTester will automatically test the performance
of your agent, will give you both the epochwise mean
reward and percentage of optimal action, an will
plot your results.

You may want to start by testing the epsilon greedy
policy with various values of epsilon, to get a grasp
of the results you are supposed to obtain.
"""

# Do not modify this class.


class AgentTester:
    def __init__(self, agentClass, N, k, iterations, params):
        self.iterations = iterations
        self.N = N
        self.agentClass = agentClass
        self.agentTable = []
        params['A'] = k
        for i in range(N):
            self.agentTable[len(self.agentTable):] = [agentClass(**params)]
        self.bandits = Banditos(self.N, k)
        self.optimal = np.argmax(self.bandits.q_stars, axis=1)

    def oneStep(self):
        rewards = np.zeros(self.N)
        optimals = np.zeros(self.N)
        for i in range(self.N):
            self.bandits.select(i)
            action = self.agentTable[i].interact()
            optimals[i] = (action == self.optimal[i]) and 1 or 0
            rewards[i] = self.bandits.act(action)
            self.agentTable[i].update(action, rewards[i])
        return rewards.mean(), optimals.mean() * 100

    def test(self):
        meanrewards = np.zeros(self.iterations)
        meanoptimals = np.zeros(self.iterations)
        for i in range(self.iterations):
            meanrewards[i], meanoptimals[i] = self.oneStep()
            display = '\repoch: {0} -- mean reward: {1} -- percent optimal: {2}'
            sys.stdout.write(display.format(i, meanrewards[i], meanoptimals[i]))
            sys.stdout.flush()
        return meanrewards, meanoptimals

# Modify only the agent class and the parameter dictionnary.

if __name__ == '__main__':
    #tester = AgentTester(ucbAgent, 2000, 10, 1000,{})
    #tester = AgentTester(softmaxAgent, 2000, 10, 1000,{"alpha":5})
    #tester = AgentTester(epsilonGreedyAgent, 2000, 10, 1000,{"epsilon":0.2})
    tester = AgentTester(epsGreedyAgent, 2000, 10, 1000,{"epsilon":0.})

    # Do not modify.
    meanrewards, meanoptimals = tester.test()
    plt.figure(1)
    plt.plot(meanrewards)
    plt.xlabel('Epoch')
    plt.ylabel('Average reward')
    plt.figure(2)
    plt.xlabel('Epoch')
    plt.ylabel('Percent optimal')
    plt.plot(meanoptimals)
    plt.show()
