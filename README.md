

<p align="center"><img align="center" src="assets/logo.png" alt="logo" style="zoom:100%;" ></p>



## Multiagent RLlib: A unified official code releasement of MARL researches made by TJU-RL-Lab

This repository contains representative research works of TJU-RL-Lab on the topic of Multiagent Reinforcement Learning.

This repository will be constantly updated to include new researches made by TJU-RL-Lab.  

(The development of this repo is in progress at present.)



## Key Features

### :rocket: Including the State-Of-The-Art

- **[API-QMIX [ICML-22 underreview]](https://arxiv.org/pdf/2203.05285.pdf)**: the state-of-the-art MARL algorithms in the [StarCraft Multi-Agent Challenge (SMAC)](https://github.com/oxwhirl/smac) benchmark. 
  - [SMAC](https://github.com/oxwhirl/smac) is [WhiRL](http://whirl.cs.ox.ac.uk/)'s environment for research in the field of collaborative multi-agent reinforcement learning (MARL) based on [Blizzard](http://blizzard.com/)'s [StarCraft II](https://en.wikipedia.org/wiki/StarCraft_II:_Wings_of_Liberty) RTS game. SMAC makes use of Blizzard's [StarCraft II Machine Learning API](https://github.com/Blizzard/s2client-proto) and [DeepMind](https://deepmind.com/)'s [PySC2](https://github.com/deepmind/pysc2) to provide a convenient interface for autonomous agents to interact with StarCraft II, getting observations and performing actions. Unlike the [PySC2](https://github.com/deepmind/pysc2), SMAC concentrates on *decentralised micromanagement* scenarios, where each unit of the game is controlled by an individual RL agent.<img src="./assets/smac.webp" alt="SMAC" style="zoom:70%;" />



## Directory Structure

| Directions                  | Sub-Directions                                              | Work (Conference)                    |
| :-------------------------- | :---------------------------------------------------------- | :----------------------------------- |
| **network_design**          | (1) action semantics; <br />(2) agent permutation invariant (equivariant) | ASN (ICLR-2020) @维埙 <br />API (ICML-2022) @晓田 |
| **credit_assignment**       |                                                             | QPD (ICML-2020)@耀东<br />Qatten  @耀东        |
| **multiagent_exploration**  |                                                             | PMIC (ICML-2022) @鹏翼                     |
| **large_scale_learning**    | (1) Game abstraction                                        | G2ANet (AAAI-2020)  @维埙                 |
| **curriculum_learning**     |                                                             | DyAN (AAAI-2020) @维埙                    |
| **hybrid_action**           |                                                             | 浩天 & 嘉顺                          |
| **self_imitation_learning** |                                                             | GASIL (AAMAS-2019) @晓田                   |
| **non-stationarity**       |                              | BPR+ (NIPS-2018) @岩哥<br /> WDDQN @岩哥<br /> DPN-BPR+(AAMAS2020) @岩哥|
| **HMARL**                   |                                                        | HIL, HCOMM, HQMIX @汤哥 |
