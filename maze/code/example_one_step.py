import pandas as pd
import numpy as np

# Example for state 9

########### INIT ##############
epsilon = 1.0 # percent of the time that agent takes a randomly selected action
state = 9 # case of the maze
direction = ["U", "R", "D", "L"]
# THETA : Grid of possible choice at each state    
theta = np.array([[np.nan, 1, np.nan, np.nan],    # S0: Start
                               [np.nan, 1, np.nan, 1],         # S1
                               [np.nan, np.nan, 1, 1],         # S2
                               [np.nan, np.nan, 1, np.nan],    # S3
                               [np.nan, 1, np.nan, np.nan],    # S4
                               [np.nan, 1, 1, 1],              # S5
                               [1, 1, np.nan, 1],              # S6
                               [1, np.nan, np.nan, 1],         # S7
                               [np.nan, np.nan, 1, np.nan],    # S8
                               [1, np.nan, 1, np.nan],         # S9
                               [np.nan, 1, 1, np.nan],         # S10
                               [np.nan, np.nan, 1, 1],         # S11
                               [1, 1, np.nan, np.nan],         # S12
                               [1, 1, np.nan, 1],              # S13
                               [1, np.nan, np.nan, 1],         # S14
                               ])      

a,b = theta.shape
Q = np.random.rand(a, b) * theta
Q_hist = Q
[m, n] = theta.shape
pi = np.zeros((m, n))

for i in range(0, m):
    pi[i, :] = theta[i, :] / np.nansum(theta[i, :])
pi = np.nan_to_num(pi) # possible choice at each state, initalise to direction / # action possible
########### END of INIT ##############



# Choose next direction based on current state (random or maximize reward, depends on epsilon)
if np.random.rand() < epsilon:
    next_direction = np.random.choice(
                    direction, p=pi[state, :])
else:
    next_direction = direction[int(np.nanargmax(Q[state, :]))]

# save action chosen based on current state
if next_direction == "U":
    action = 0
elif next_direction == "R":
    action = 1
elif next_direction == "D":
    action = 2
elif next_direction == "L":
    action = 3



# directly derives from the formula of Q-learning
# eta = learning_rate, by default 0.15 here
# S : State
# A : Action
# Q(S,A) = Q(S,A) + learning_rate * (reward + discout_factor * max(S+1, A+1) - Q(S,A))
# s_next : state depending on the variable *next_direction* chosen
# a_next : action that would be chosen on the next_state
s = state
a = action
eta = 0.15
gamma = 0.9

# Get next_state base on next_direction chosen
if next_direction == "U":
    s_next = state - 4
elif next_direction == "R":
    s_next = state + 1
elif next_direction == "D":
    s_next = state + 4
elif next_direction == "L":
    s_next = state - 1

# get next_action based on next_state
if np.random.rand() < epsilon:
    next_next_direction = np.random.choice(
                    direction, p=pi[s_next, :])
else:
    next_next_direction = direction[int(np.nanargmax(Q[s_next, :]))]
    
if next_next_direction == "U":
    a_next = 0
elif next_next_direction == "R":
    a_next = 1
elif next_next_direction == "D":
    a_next = 2
elif next_next_direction == "L":
    a_next = 3

# reward function, easy in this case, we give reward only when goal is reached (state 15)
r = 1 if s_next == 15 else 0

# we update the q variable (with the Q-Learning function)
Q[s, a] = Q[s, a] + eta * (r + gamma * np.nanmax(Q[s_next, :]) - Q[s, a])

# solve_maze
# idea is to redo the last steps until we reach state_15