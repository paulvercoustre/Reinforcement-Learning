
import sys

import pylab as plb
import numpy as np
import mountaincar
import itertools as iter


class RandomAgent():
    def __init__(self):
        """
        Initialize your internal state
        """
        pass

    def act(self):
        """
        Choose action depending on your internal state
        """
        return np.random.randint(-1, 2)

    def update(self, next_state, reward):
        """
        Update your internal state
        """
        pass

# implement your own agent here

def kernel(x_Vx, s_ij, p, k): # kernel function
    phi_value = {}
    for i,j in iter.product(xrange(p+1),xrange(k+1)):
        phi_value[i,j] = np.exp(-(x_Vx[0]-s_ij[i,j][0])**2) * np.exp(-(x_Vx[1]-s_ij[i,j][1])**2)
    return phi_value
        
def dict_to_array(dict_input, p, k): # converts dictionnary to array to compute dot product
    iterator = 0
    phi_as_array = np.ones((p+1) * (k+1))
    for i, j in iter.product(xrange(p+1),xrange(k+1)):       
        phi_as_array[iterator] = dict_input[i,j]
        iterator += 1
    return phi_as_array
        
class QlearningAgent():
    def __init__(self):
        self.alpha = 0.1
        self.gamma = 0.4
        self.epsilon = 0.1 # exploration/explotation ratio
        self.Q = np.zeros(3) # the Q values
        #self.phi_array = np.zeros((p+1) * (k+1)) # the representation vector
        
        self.p = 75 # granularity of discretisation for spacial coordinate
        self.k = 20 # granularity of discretisation for horizontal velocity
        self.s = {}
        for i, j in iter.product(xrange(self.p+1),xrange(self.k+1)):
            self.s[i,j] = np.array([-150 + i * 150/self.p, -20 + j * 40/self.k])
    
        self.weight = np.ones((3,(self.p+1) * (self.k+1))) / ((self.p+1) * (self.k+1)) # set equal weights initially
        self.x = np.random.uniform(-130, -50)
        self.vx = np.random.uniform(-5, -5)
        self.current_state = [self.x, self.vx] # the initial state of the car 
        
    def act(self):
        self.phi_dict = kernel(self.current_state, self.s, self.p, self.k)
        self.phi_array = dict_to_array(self.phi_dict, self.p, self.k)
        rand = np.random.uniform()
        if rand < self.epsilon:
            self.action = np.random.randint(-1, 2) # hardcoded value !
        else: 
            # greedy case. Take action with the maximal Q-value
            self.Q = np.dot(self.weight,self.phi_array)
            self.action = np.argmax(self.Q) - 1
        return self.action
        
    def update(self, next_state, reward):
        self.next_phi_dict = kernel(next_state, self.s, self.p, self.k)
        self.next_phi_array = dict_to_array(self.next_phi_dict, self.p, self.k)
        self.next_Q = np.dot(self.weight,self.next_phi_array)
        
        self.weight[self.action + 1,:] += self.alpha * (reward + 
                   self.gamma * max(self.next_Q) - self.Q[self.action + 1]) * self.phi_array
        self.current_state = next_state

# test class, you do not need to modify this class
class Tester:

    def __init__(self, agent):
        self.mountain_car = mountaincar.MountainCar()
        self.agent = agent

    def visualize_trial(self, n_steps=200):
        """
        Do a trial without learning, with display.

        Parameters
        ----------
        n_steps -- number of steps to simulate for
        """

        # prepare for the visualization
        plb.ion()
        mv = mountaincar.MountainCarViewer(self.mountain_car)
        mv.create_figure(n_steps, n_steps)
        plb.draw()

        # make sure the mountain-car is reset
        self.mountain_car.reset()

        for n in range(n_steps):
            print('\rt =', self.mountain_car.t)
            print("Enter to continue...")
            raw_input()

            sys.stdout.flush()

            reward = self.mountain_car.act(self.agent.act())
            self.agent.state = [self.mountain_car.x, self.mountain_car.vx]

            # update the visualization
            mv.update_figure()
            plb.draw()

            # check for rewards
            if reward > 0.0:
                print("\rreward obtained at t = ", self.mountain_car.t)
                break

    def learn(self, n_episodes, max_episode):
        """
        params:
            n_episodes: number of episodes to perform
            max_episode: maximum number of steps on one episode, 0 if unbounded
        """

        rewards = np.zeros(n_episodes)
        for c_episodes in range(1, n_episodes):
            self.mountain_car.reset()
            step = 1
            while step <= max_episode or max_episode <= 0:
                reward = self.mountain_car.act(self.agent.act())
                self.agent.update([self.mountain_car.x, self.mountain_car.vx],
                                  reward)
                rewards[c_episodes] += reward
                if reward > 0.:
                    break
                step += 1
            formating = "end of episode after {0:3.0f} steps,\
                           cumulative reward obtained: {1:1.2f}"
            print(formating.format(step-1, rewards[c_episodes]))
            sys.stdout.flush()
        return rewards


if __name__ == "__main__":
    # modify RandomAgent by your own agent with the parameters you want
    #agent = RandomAgent()
    agent = QlearningAgent()
    test = Tester(agent)
    # you can (and probably will) change these values, to make your system
    # learn longer
    test.learn(10, 1000)

    print("End of learning, press Enter to visualize...")
    raw_input()
    test.visualize_trial()
    plb.show()
