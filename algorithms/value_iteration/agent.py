import numpy as np

from agent import BaseAgent


class Agent(BaseAgent):

    def __init__(self, grid, obey_prob, gamma=0.9, min_delta=1e-3):
        super().__init__(grid, obey_prob, gamma)
        self.min_delta = min_delta

        self.V = {}
        states = self.env.all_states
        for s in states:
            self.V[s] = 0

        self.policy = {}
        for s in self.env.non_terminal_states:
            self.policy[s] = np.random.choice(self.env.all_possible_actions)

        self.i, self.j = None, None

    def set_state(self, state):
        self.i, self.j = state

    @property
    def current_state(self):
        return (self.i, self.j)

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

    def check_move(self, action):
        i = self.i
        j = self.j
        # check if the move is legal
        if action in self.env.actions[(self.i, self.j)]:
            if action == 'U':
                i -= 1
            elif action == 'D':
                i += 1
            elif action == 'R':
                j += 1
            elif action == 'L':
                j -= 1
        # return a reward
        reward = self.env.rewards.get((i, j), 0)
        return ((i, j), reward)

    def get_transition_probs(self, action):
        # returns a list of (probability, reward, s') transition tuples
        probs = []
        state, reward = self.check_move(action)
        probs.append((self.obey_prob, reward, state))
        disobey_prob = 1 - self.obey_prob
        if not (disobey_prob > 0.0):
            return probs
        if action == 'U' or action == 'D':
            state, reward = self.check_move('L')
            probs.append((disobey_prob / 2, reward, state))
            state, reward = self.check_move('R')
            probs.append((disobey_prob / 2, reward, state))
        elif action == 'L' or action == 'R':
            state, reward = self.check_move('U')
            probs.append((disobey_prob / 2, reward, state))
            state, reward = self.check_move('D')
            probs.append((disobey_prob / 2, reward, state))
        return probs

    def solve(self):
        self.calculate_values()
        self.calculate_greedy_policy()

    def calculate_values(self):
        # repeat until convergence
        # V[s] = max[a]{ sum[s',r] { p(s',r|s,a)[r + gamma*V[s']] } }
        while True:
            biggest_change = 0
            for s in self.env.non_terminal_states:
                old_v = self.V[s]
                _, new_v = self.best_action_value(s)
                self.V[s] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - new_v))

            if biggest_change < self.min_delta:
                break
        return self.V

    def best_action_value(self, s):
        # finds the highest value action (argmax_a(V)) from state s, returns the action and value
        best_a = None
        best_value = float('-inf')
        self.set_state(s)
        # loop through all possible actions to find the best current action
        for a in self.env.all_possible_actions:
            transitions = self.get_transition_probs(a)
            expected_v = 0
            expected_r = 0
            for (prob, r, state_prime) in transitions:
                expected_r += prob * r
                expected_v += prob * self.V[state_prime]
            v = expected_r + self.gamma * expected_v
            if v > best_value:
                best_value = v
                best_a = a
        return best_a, best_value

    def calculate_greedy_policy(self):
        for s in self.policy.keys():
            self.set_state(s)
            # loop through all possible actions to find the best current action
            best_a, _ = self.best_action_value(s)
            self.policy[s] = best_a
        return self.policy

    def print_solution(self):
        print("Rewards:")
        self.print_rewards()
        print("Values:")
        self.print_values()
        print("Policy:")
        self.print_policy()

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