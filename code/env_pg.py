'''
    env_pg.py: the environment for policy-gradient learning.
'''

# (hyper-parameters) reward values
reward_success = 10000  # reward for success
reward_city = 200       # reward coefficient for reaching a city
reward_normal = -1      # reward for a normal step
reward_out = -100       # reward while out of the bounds
reward_repeat = -100    # reward for visiting a city twice

# some constants
actions = [0, 1, 2, 3]  # four action values
dir_X = [0, 0, -1, 1]   # change in X corresponding to the action
dir_Y = [-1, 1, 0, 0]   # change in Y corresponding to the action
S_X = [1, 2, 3, 4, 5, 5, 7] # X coordinates for cities
S_Y = [2, 6, 2, 0, 3, 9, 7] # Y coordinates for cities

# the Env class
class Env:

    # constructor
    def __init__(self):
        self.S_visited = 0  # record visited cities
        self.X_pos = 0      # the current X
        self.Y_pos = 0      # the current Y
        self.city_cnt = 0   # the number of visited cities
        self.record = [[0 for _ in range(10)] for __ in range(10)]  # record the time visited for each coordinate
        self.step_cnt = 0   # the current step

    # resetter
    def reset(self):
        self.S_visited = 0
        self.X_pos = 0
        self.Y_pos = 0
        self.city_cnt = 0
        self.record = [[0 for _ in range(10)] for __ in range(10)]
        self.step_cnt = 0

    # function for each step
    # returns self, reward, done
    #   done = 0 for a normal step
    #   done = 1 for success
    #   done = 2 while out-of-bounds
    #   done = 3 for visiting a city twice
    #   done = 4 for a probable endless loop
    def step(self, action):

        #update the coordinates and the cnt
        self.X_pos += dir_X[action]
        self.Y_pos += dir_Y[action]
        self.step_cnt += 1

        # succeed, terminate
        if self.S_visited == 0b1111111 and self.X_pos == 0 and self.Y_pos == 0:
            return self, reward_success, 1
        
        # the start city visited twice, terminate
        elif self.X_pos == 0 and self.Y_pos == 0:
            return self, reward_repeat, 3
        
        # out of the bounds, terminate
        elif self.X_pos < 0 or self.X_pos > 9 or self.Y_pos < 0 or self.Y_pos > 9:
            return self, reward_out, 2
        
        # a probable endless loop, terminate
        elif self.record[self.X_pos][self.Y_pos] > 4:
            return self, 2*reward_normal, 4
        
        else:
            for _ in range(7):
                if self.X_pos == S_X[_] and self.Y_pos == S_Y[_]:

                    # a city visited twice, terminate
                    if (self.S_visited>>_)&1 == 1:
                        return self, reward_repeat, 3

                    # reaching a new city, continue with a squared-rated reward
                    else:
                        self.S_visited += 1<<_
                        self.city_cnt += 1
                        return self, reward_city*self.city_cnt*self.city_cnt, 0
            
            # a normal step
            self.record[self.X_pos][self.Y_pos] += 1
            return self, reward_normal, 0

