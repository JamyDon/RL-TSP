'''
    q_learning.py: the process of Q-learning
'''

import env_ql
import numpy as np
import matplotlib.pyplot as plt
import random
from cmath import inf

# hyper-parameters
episode = 10000000  # total episodes
epsilon = 0.1       # for epsilon greedy
alpha = 0.01        # learning factor
gamma = 0.9         # discount factor

# some constants
actions = [0, 1, 2, 3]  # four action values

# calculate the corresponding coordinate of the Q-table
def get_pos(S_visited, X_pos, Y_pos):
    return S_visited*100 + X_pos*10 + Y_pos

# the main process of Q-learning
def q_learning():

    # initialization
    random.seed(0)
    Q = [[-1 for _ in range(4)] for __ in range (100*0b10000000)]   # Q is initialized as -1
    Q = np.array(Q)
    env = env_ql.Env()

    # data for the plot
    r = []

    # the main iteration process
    for i in range(episode):

        # reset the variations
        env.reset()
        S_visited = 0b0000000
        X_pos = 0
        Y_pos = 0

        # one episode
        while True:
            state_pos = get_pos(S_visited, X_pos, Y_pos)

            # epsilon-greedy
            rand = random.random()
            if rand < epsilon:
                action = random.choice(actions)
            else:
                Q_max = -10000
                for _ in range(4):
                    if Q[state_pos][_] > Q_max:
                        Q_max = Q[state_pos][_]
                        action = actions[_] # choose the max Q
            
            # interact with the environment
            new_env, reward, done = env.step(action)
            S_visited = new_env.S_visited
            X_pos = new_env.X_pos
            Y_pos = new_env.Y_pos

            # update the Q-table
            if done == 2:   # out of the bounds
                Q[state_pos][action] = (1-alpha)*Q[state_pos][action] + alpha*reward
                break
            new_state_pos = get_pos(S_visited, X_pos, Y_pos)
            V_next = -10000
            for _ in range(4):
                if Q[new_state_pos][_] > V_next:
                    V_next = Q[new_state_pos][_]
            Q[state_pos][action] = (1-alpha)*Q[state_pos][action] + alpha*(reward + gamma*V_next)
            if done != 0:
                break

        # update the data for the plot
        if (i+1) % 10000 == 0:

            # initialization
            env.reset()
            S_visited = 0b0000000
            X_pos = 0
            Y_pos = 0
            step = 0
            total_reward = 0

            # one episode
            while True:
                step += 1
                state_pos = get_pos(S_visited, X_pos, Y_pos)

                # select action with the max Q value
                Q_max = -10000
                for _ in range(4):
                    if Q[state_pos][_] > Q_max:
                        Q_max = Q[state_pos][_]
                        action = actions[_]
                
                # interact with the environment
                new_env, reward, done = env.step(action)
                S_visited = new_env.S_visited
                X_pos = new_env.X_pos
                Y_pos = new_env.Y_pos
                total_reward += reward

                if done == 1:
                    break
                elif done != 0:
                    total_reward = 0
                    break
                if new_env.step_cnt >= 100:
                    total_reward = 0
                    break
            
            # update the r list
            r.append(total_reward)

    # save the Q-table
    np.save('../result/data/q_table', Q)

    # generate the plot
    x = range(1000)
    plt.figure(figsize = (12, 8), dpi = 80)
    plt.plot(x, r)
    plt.xlabel('Episode(10k)')
    plt.ylabel('Reward')
    plt.title('Curve of Q-learning')
    plt.savefig('../result/figure/q_learning.png')

    # show the final results of Q-learning
    print('-----------------q_learning------------------')
    env.reset()
    S_visited = 0b0000000
    X_pos = 0
    Y_pos = 0
    step = 0
    while True:
        step += 1
        state_pos = get_pos(S_visited, X_pos, Y_pos)

        Q_max = -10000
        for _ in range(4):
            if Q[state_pos][_] > Q_max:
                Q_max = Q[state_pos][_]
                action = actions[_]
        
        new_env, reward, done = env.step(action)
        S_visited = new_env.S_visited
        X_pos = new_env.X_pos
        Y_pos = new_env.Y_pos

        print('Step ', step, ': [', X_pos, ',', Y_pos, ']', sep='')
        if step >= 100:
            print('-------------------------------------------')
            print("Exception 4")
            print('-------------------------------------------')
            break
        
        if done == 1:
            print('-------------------------------------------')
            print('Succeed at step', step)
            print('-------------------------------------------')
            break
        if done != 0:
            print('-------------------------------------------')
            print('Exception ', done, sep='')
            print('-------------------------------------------')
            break
