import numpy as np


class Agent:

    def __init__(self, strategy="q"):
        self.strategy = strategy
        self.state = 0
        self.pi = None
        self.action = None
        self.state_history = [[0, np.nan]]
        self.step_log = list()
        # Represent the move that the agent can do at each state
        # ex : at state2, the agent can go right or left
        # Structure : [Up, Right, Down, Left]
        self.theta = np.array([[np.nan, 1, np.nan, np.nan],    # S0: Start
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
                               ])                              # S15: End
        if strategy in ["q"]:
            a, b = self.theta.shape
            self.epsilon = 1.0
            self.Q = np.random.rand(a, b) * self.theta
        else:
            raise ValueError("undefined strategy {}".format(self.strategy))
        self.initial_covert_from_theta_to_pi()

    def initial_covert_from_theta_to_pi(self):

        [m, n] = self.theta.shape
        pi = np.zeros((m, n))

        if self.strategy in ["q"]:
            # We devide each move by the # of possibilities
            # If you can go left, down, or up then you divide each element by 3
            # State_i = [0.3333,0,0.3333,0.3333]
            for i in range(0, m):
                pi[i, :] = self.theta[i, :] / np.nansum(self.theta[i, :])
        else:
            raise ValueError("undefined strategy {}".format(self.strategy))
        self.pi = np.nan_to_num(pi)

    def get_action(self):
        direction = ["U", "R", "D", "L"]

        if self.strategy in ["q"]:
            # this condition directly derives from the greedy search logic
            # epsilon : percent of the time that agent takes a randomly selected action
            # initialised at 1 (meaning that it will take a random choice at first)
            # state0: only one choice, to go right
            if np.random.rand() < self.epsilon:
                next_direction = np.random.choice(
                    direction, p=self.pi[self.state, :])
            
            # case where the choice on the direction is not random, it goes directly to the direction
            # with the most reward
            else:
                next_direction = direction[int(np.nanargmax(self.Q[self.state, :]))]
        else:
            raise ValueError("undefined strategy {}".format(self.strategy))

        if next_direction == "U":
            action = 0
        elif next_direction == "R":
            action = 1
        elif next_direction == "D":
            action = 2
        elif next_direction == "L":
            action = 3
        else:
            raise ValueError("undefined direction {}".format(next_direction))
        return action

    def move_next_state(self, action=None):
        """Returns next states for given policy and state"""
        direction = ["U", "R", "D", "L"]

        if action is not None:
            next_direction = direction[action]
        else:
            next_direction = np.random.choice(
                direction, p=self.pi[self.state, :])

        if next_direction == "U":
            s_next = self.state - 4
        elif next_direction == "R":
            s_next = self.state + 1
        elif next_direction == "D":
            s_next = self.state + 4
        elif next_direction == "L":
            s_next = self.state - 1
        else:
            raise ValueError("Unknown Direction %s" % next_direction)

        self.state = s_next

    def update_Q(self, s, a, r, s_next, a_next, Q, eta, gamma):
        """Update Q for 'q' strategy"""
        if self.strategy in ['q']:
            if s_next == 15:
                Q[s, a] = Q[s, a] + eta * (r - Q[s, a])
            else:
                # directly derives from the formula of Q-learning
                # eta = learning_rate, by default 0.15 here
                # S : State
                # A : Action
                # Q(S,A) = Q(S,A) + learning_rate * (reward + discout_factor * max(S+1, A+1) - Q(S,A))
                # s_next : state depending on the variable *next_direction* chosen
                # a_next : action that would be chosen on the next_state
                Q[s, a] = Q[s, a] + eta * \
                    (r + gamma * np.nanmax(Q[s_next, :]) - Q[s, a])
        else:
            raise ValueError("undefined strategy {}".format(self.strategy))
        self.Q = Q

    def solve_maze(self, eta=None, gamma=None):
        a_next = self.get_action()
        while True:
            a = a_next
            self.state_history[-1][1] = a
            s = self.state
            self.move_next_state(action=a)
            s_next = self.state
            self.state_history.append([s_next, np.nan])

            if s_next == 15:
                r = 1
                a_next = np.nan
            else:
                r = 0
                a_next = self.get_action()

            self.update_Q(s, a, r, s_next, a_next, self.Q, eta, gamma)

            if s_next == 15:
                break

    def train(self, stop_epsilon=10**-4, eta=0.15, gamma=0.9, tot_episode=100):

        v = np.nanmax(self.Q, axis=1) # Get a vector of max prob at each state
        episode = 1
        V = list()
        V.append(v)
        while True:
            print("Episode: ", str(episode))
            self.epsilon = self.epsilon / 2 # Each iteration we divide the epsilon by 2 (making the agent less greedy)
            self.solve_maze(eta=eta, gamma=gamma) # Solve maze (update Q)
            new_v = np.nanmax(self.Q, axis=1) # new probability vector for each state (updated with last solution)
            print("State value difference: ", np.sum(np.abs(new_v - v)))
            v = new_v
            V.append(v)
            print("Complete in %d steps" % (len(self.state_history) - 1))
            episode += 1
            self.step_log.append(len(self.state_history) - 1)
            if episode > tot_episode:
                print("State value after training: ", V[-1])
                break
            self.state = 0 # reset state for next iteration
            self.state_history = [[0, np.nan]]

    def __repr__(self):
        return "Agent: {}".format(self.strategy)