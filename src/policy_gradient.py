'''
    policy_gradient.py: the process of policy-gradient learning
'''

import env_pg
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from cmath import inf

# hyper-parameters
restart = 100       # restart times
epoch = 20          # total epoches
batch_size = 300    # batch size for each epoch
alpha = 0.001       # step size
gamma = 0.9         # discount rate

# some constants
actions = [0, 1, 2, 3]  # four action values

# calculate the corresponding coordinate of the Q-table
def get_pos(S_visited, X_pos, Y_pos):
    return S_visited*100 + X_pos*10 + Y_pos

# choose the action and calculate the gradients using softmax
def softmax(x_1, x_2, x_3, x_4):
    max_x = max(x_1, x_2, x_3, x_4)
    ex_1 = math.exp(x_1 - max_x)
    ex_2 = math.exp(x_2 - max_x)
    ex_3 = math.exp(x_3 - max_x)
    ex_4 = math.exp(x_4 - max_x)
    sum_ex = ex_1 + ex_2 + ex_3 + ex_4

    # calculate the gradients
    gradient = []
    gradient.append(-ex_1/sum_ex)
    gradient.append(-ex_2/sum_ex)
    gradient.append(-ex_3/sum_ex)
    gradient.append(-ex_4/sum_ex)

    # choose the action with the probability after softmax
    p_1 = ex_1 / sum_ex
    p_2 = ex_2 / sum_ex + p_1
    p_3 = ex_3 / sum_ex + p_2
    rand = random.random()
    if rand < p_1:
        action = 0
        gradient[0] += 1
    elif rand < p_2:
        action = 1
        gradient[1] += 1
    elif rand < p_3:
        action = 2
        gradient[2] += 1
    else:
        action = 3
        gradient[3] += 1

    return action, gradient

# the main process of policy-gradient learning
# returns final_step, parameter, r
# - final_step is the step cost in the final test
# - parameter is the final parameter matrix
# - r is the r list for the plot
def policy_gradient_learning():

    # initialization
    global alpha
    parameter = [[2*random.random(), 2*random.random(), 2*random.random(), 2*random.random()] for __ in range (100*0b10000000)]
    parameter = np.array(parameter)
    env = env_pg.Env()
    final_step = 0

    # data for the plot
    r = []

    # each epoch
    for i in range(epoch):

        # initialize the delta_parameter
        delta_parameter = [[float(0) for _ in range(4)] for __ in range (100*0b10000000)]
        delta_parameter = np.array(delta_parameter)

        # each episode
        for j in range(batch_size):

            # reset the variations
            env.reset()
            S_visited = 0b0000000
            X_pos = 0
            Y_pos = 0
            states = []
            gradients = []
            rewards = []

            # one episode
            while True:
                state_pos = get_pos(S_visited, X_pos, Y_pos)
                states.append(state_pos)

                # choose action and calculate gradients using softmax
                action, gradient = softmax(parameter[state_pos][0],
                                           parameter[state_pos][1],
                                           parameter[state_pos][2],
                                           parameter[state_pos][3])
                
                # interact with the environment
                new_env, reward, done = env.step(action)
                S_visited = new_env.S_visited
                X_pos = new_env.X_pos
                Y_pos = new_env.Y_pos

                # update the backward memory
                gradients.append(gradient)
                rewards.append(reward)
            
                if done == 0:
                    continue

                # if done, update delta_parameter
                else:
                    pos = len(states) - 1
                    rewards.append(0)
                    while pos >= 0:
                        rewards[pos] += gamma * rewards[pos + 1]
                        for act in range(4):
                            delta_parameter[states[pos]][act] += float(alpha*rewards[pos]*gradients[pos][act])
                        pos -= 1
                    break
        
        # the whole epoch done, update the parameter-matrix
        for _ in range(100*0b10000000):
            for __ in range(4):
                parameter[_][__] += delta_parameter[_][__]

        # test the parameters for the plot
        env.reset()
        S_visited = 0b0000000
        X_pos = 0
        Y_pos = 0
        step = 0
        total_reward = 0
        while True:
            step += 1
            state_pos = get_pos(S_visited, X_pos, Y_pos)

            para_max = parameter[state_pos][0]
            action = 0
            for _ in range(4):
                if parameter[state_pos][_] > para_max:
                    para_max = parameter[state_pos][_]
                    action = _
            
            new_env, reward, done = env.step(action)
            S_visited = new_env.S_visited
            X_pos = new_env.X_pos
            Y_pos = new_env.Y_pos
            total_reward += reward

            if total_reward < -10000:
                final_step = inf
                break
            
            if done != 0:
                if done == 1:
                    final_step = step
                    break
                final_step = inf
                break
        
        # update the r list
        r.append(total_reward)
    
    return final_step, parameter, r

# the main process
def policy_gradient():

    # initialization
    random.seed(0)
    best_step = inf
    best_parameter = [[0.0 for _ in range(4)] for __ in range(100*0b10000000)]
    best_r = []

    # restart the process in order to get the global best
    for _ in range(restart):

        # do one process
        final_step, parameter, r = policy_gradient_learning()

        # update the best parameter
        if best_step > final_step:
            best_step = final_step
            best_parameter = parameter
            best_r = r
    
    # save the best parameter-matrix
    np.save('../result/data/parameter_matrix', best_parameter)

    # generate the plot
    x = range(epoch)
    plt.figure(figsize = (12, 8), dpi = 80)
    plt.plot(x, best_r)
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.title('Curve of Policy-gradient Learning')
    plt.savefig('../result/figure/policy_gradient.png')

    # show the final results    
    env = env_pg.Env()
    print('-------------policy_gradient---------------')
    env.reset()
    S_visited = 0b0000000
    X_pos = 0
    Y_pos = 0
    step = 0
    while True:
        step += 1
        state_pos = get_pos(S_visited, X_pos, Y_pos)

        para_max = best_parameter[state_pos][0]
        action = 0
        for _ in range(4):
            if best_parameter[state_pos][_] > para_max:
                para_max = best_parameter[state_pos][_]
                action = _
        
        new_env, reward, done = env.step(action)
        S_visited = new_env.S_visited
        X_pos = new_env.X_pos
        Y_pos = new_env.Y_pos

        print('Step ', step, ': [', X_pos, ',', Y_pos, ']', sep='')
        if step >= 100:
            print('--------------------------------------------')
            print("Exception 100")
            print('--------------------------------------------')
            break
        
        if done == 1:
            print('--------------------------------------------')
            print('Succeed at step', step)
            print('--------------------------------------------')
            break
        if done != 0:
            print('--------------------------------------------')
            print('Exception ', done, sep='')
            print('--------------------------------------------')
            break
