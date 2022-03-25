# Non-stationarity  (especially for independent learners)


<p align="center"><img align="center" src="../assets/non-stationary.png" alt="non-stationary" style="zoom:80%;" /></p>



In Markov games, the state transition function and the reward function of each agent depend on the actions of all agents. During the training of multiple agents, the policy of each agent changes through time. As a result, each agents’ perceived transition and reward functions change as well, which causes the environment faced by each individual agent to be non-stationary and breaks the Markov assumption that governs the convergence of most single-agent RL algorithms. In the worst case, each agent can enter an endless cycle of adapting to other agents. 
