import numpy as np

from agent import BaseAgent


class Agent(BaseAgent):

    def __init__(self, grid, obey_prob, gamma=0.9, epsilon=0.0):
        super().__init__(grid, obey_prob, gamma)
        self.epsilon = epsilon

        self.V = {}
        states = self.env.all_states
        for s in states:
            self.V[s] = 0

        self.policy = {}
        for s in self.env.actions.keys():
            self.policy[s] = np.random.choice(self.env.all_possible_actions)

        # initialize Q(s,a) and returns
        self.Q = {}
        self.returns = {} # dictionary of state -> list of returns we've received
        states = self.env.non_terminal_states
        for s in states:
            self.Q[s] = {}
            for a in self.env.all_possible_actions:
                self.Q[s][a] = 0
                self.returns[(s,a)] = []

        self.i, self.j = None, None

    def set_state(self, state):
        self.i, self.j = state

    @property
    def current_state(self):
        return (self.i, self.j)

    def epsilon_action(self, state):
        action = self.policy[state]
        p = np.random.random()
        if p < 1 - self.epsilon:
            return action
        return np.random.choice(self.env.all_possible_actions)

    def stochastic_move(self, action):
        p = np.random.random()
        if p <= self.obey_prob:
            return action
        if action == 'U' or action == 'D':
            return np.random.choice(['L', 'R'])
        elif action == 'L' or action == 'R':
            return np.random.choice(['U', 'D'])
    
    def move(self, action):
        actual_action = self.stochastic_move(action)
        if actual_action in self.env.actions[(self.i, self.j)]:
            if actual_action == 'U':
                self.i -= 1
            elif actual_action == 'D':
                self.i += 1
            elif actual_action == 'R':
                self.j += 1
            elif actual_action == 'L':
                self.j -= 1
        return self.env.rewards.get((self.i, self.j), 0)

    def play_game(self):
        # returns a list of states and corresponding returns
        s = (2, 0)
        self.set_state(s)
        a = self.epsilon_action(s)

        # keep in mind that reward is lagged by one time step
        # r(t) results from taking action a(t-1) from s(t-1) and landing in s(t)
        states_actions_rewards = [(s, a, 0)]
        while True:
            r = self.move(a)
            s = self.current_state
            if self.env.game_over(s):
                states_actions_rewards.append((s, None, r))
                break
            else:
                a = self.epsilon_action(s)
                states_actions_rewards.append((s, a, r))

        # calculate the returns by working backwards from the terminal state
        G = 0
        states_actions_returns = []
        first = True
        for s, a, r in reversed(states_actions_rewards):
            # a terminal state has a value of 0 by definition
            # this is the first state we encounter in the reversed list
            # we'll ignore its return (G) since it doesn't correspond to any move
            if first:
                first = False
            else:
                states_actions_returns.append((s, a, G))
            G = r + self.gamma*G
        states_actions_returns.reverse() # back to the original order of states visited
        return states_actions_returns

    def solve(self, n_episodes=10000):
        self.calculate_policy(n_episodes)
        self.calculate_values()

    def calculate_policy(self, n_episodes):
        # repeat for the number of episodes specified (enough that it converges)
        for t in range(n_episodes):
            if t % 1000 == 0:
                print(t)
            # generate an episode using the current policy
            biggest_change = 0
            states_actions_returns = self.play_game()
            # calculate Q(s,a)
            seen_state_action_pairs = set()
            for s, a, G in states_actions_returns:
                # check if we have already seen s
                # first-visit Monte Carlo optimization
                sa = (s, a)
                if sa not in seen_state_action_pairs:
                    self.returns[sa].append(G)
                    old_q = self.Q[s][a]
                    # the new Q[s][a] is the sample mean of all our returns for that (state, action)
                    self.Q[s][a] = np.mean(self.returns[sa])
                    biggest_change = max(biggest_change, np.abs(old_q - self.Q[s][a]))
                    seen_state_action_pairs.add(sa)
            # calculate new policy pi(s) = argmax[a]{ Q(s,a) }
            for s in self.policy.keys():
                a = max(self.Q[s], key=self.Q[s].get)
                self.policy[s] = a
        return self.policy
        
    def calculate_values(self):
        # calculate values for each state (just to print and compare)
        # V(s) = max[a]{ Q(s,a) }
        self.V = {}
        for s in self.policy.keys():
            a = max(self.Q[s], key=self.Q[s].get)
            self.V[s] = self.Q[s][a]
        return self.V

    def print_solution(self):
        print("Rewards:")
        self.print_rewards()
        print("Values:")
        self.print_values()
        print("Policy:")
        self.print_policy()

    def max_dict(self, d):
        # returns the argmax (key) and max (value) from a dictionary
        # put this into a function since we are using it so often
        max_key = None
        max_val = float('-inf')
        for k, v in d.items():
            if v > max_val:
                max_val = v
                max_key = k
        return max_key, max_val

    def print_rewards(self):
        for i in range(self.env.width):
            print("---------------------------")
            for j in range(self.env.height):
                r = self.env.rewards.get((i,j), 0)
                if r >= 0:
                    print(" %.2f|" % r, end="")
                else:
                    print("%.2f|" % r, end="")
            print("")

    def print_values(self):
        for i in range(self.env.width):
            print("---------------------------")
            for j in range(self.env.height):
                v = self.V.get((i,j), 0)
                if v >= 0:
                    print(" %.2f|" % v, end="")
                else:
                    print("%.2f|" % v, end="")
            print("")

    def print_policy(self):
        for i in range(self.env.width):
            print("---------------------------")
            for j in range(self.env.height):
                a = self.policy.get((i,j), ' ')
                print("  %s  |" % a, end="")
            print("")