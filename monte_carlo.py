from grid_world import standard_grid
from algorithms.monte_carlo.agent import Agent


N_EPISODES = 2000


if __name__ == '__main__':
    grid = standard_grid(step_cost=-0.1)
    agent = Agent(grid=grid, obey_prob=0.8, gamma=0.9, epsilon=0.2)

    agent.solve(n_episodes=N_EPISODES)
    agent.print_solution()
