

class Grid:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def set(self, rewards, actions):
        self.rewards = rewards
        self.actions = actions

    @property
    def all_possible_actions(self):
        return ('U', 'D', 'L', 'R')

    @property
    def non_terminal_states(self):
        return self.actions.keys()

    @property
    def all_states(self):
        return set(self.actions.keys()) | set(self.rewards.keys())


def standard_grid(step_cost=None):
    # define a grid that describes the reward for arriving at each state
    # and possible actions at each state
    # the grid looks like this
    # x means you can't go there
    # number means reward at that state
    # .  .  .  1
    # .  x  . -1
    # s  .  .  .
    # step_cost (float): a penalty applied each step to minimize the number of moves
    g = Grid(3, 4)
    rewards = {(0, 3): 1, (1, 3): -1}
    actions = {
    (0, 0): ('D', 'R'),
    (0, 1): ('L', 'R'),
    (0, 2): ('L', 'D', 'R'),
    (1, 0): ('U', 'D'),
    (1, 2): ('U', 'D', 'R'),
    (2, 0): ('U', 'R'),
    (2, 1): ('L', 'R'),
    (2, 2): ('L', 'R', 'U'),
    (2, 3): ('L', 'U'),
    }
    g.set(rewards, actions)
    if step_cost is not None:
        g.rewards.update({
            (0, 0): step_cost,
            (0, 1): step_cost,
            (0, 2): step_cost,
            (1, 0): step_cost,
            (1, 2): step_cost,
            (2, 0): step_cost,
            (2, 1): step_cost,
            (2, 2): step_cost,
            (2, 3): step_cost,
        })
    return g