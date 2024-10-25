# Reinfocement Learning

## Reinforcement Learning in Tic-Tac-Toe
### Description
This repository presents the implementation of a Reinforcement Learning agent to play Tic-Tac-Toe game against a human player. The project was developed based on Stanford classes available on YouTube, with the goal of applying theoretical concepts of reinforcement learning in a practical and interactive scenario.

### Main Techniques Implemented
SARSA (State-Action-Reward-State-Action): An on-policy learning algorithm that updates Q values ​​based on the actions actually taken by the agent.
Epsilon-Greedy Policy: Used to balance the exploration of new actions and the exploration of actions known to be the best.
Q-Table: Data structure to store the Q values ​​of state-action pairs, allowing the agent to learn the best game policy.
Discount Factor and Learning Rate: Configurable to control the importance of future rewards and the agent's learning speed.
End State and Draw Management: Functions to determine when the game ends, either by win or draw, adjusting rewards accordingly.
