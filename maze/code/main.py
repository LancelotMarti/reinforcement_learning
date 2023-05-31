from agent import Agent
from maze import Maze

agent = Agent('q') # q-learning strategy
agent.train()

maze = Maze()
maze.save_animation('maze.gif', agent.state_history)