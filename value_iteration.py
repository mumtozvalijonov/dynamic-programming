from agent import Agent
from grid_world import standard_grid


if __name__ == '__main__':
    grid = standard_grid(step_cost=-0.1)
    agent = Agent(grid=grid, obey_prob=0.8, gamma=0.9)

    agent.solve()
    agent.print_solution()
