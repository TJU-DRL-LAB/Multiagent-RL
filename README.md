<p align="center"><img align="center" src="assets/logo.png" alt="logo" style="zoom:100%;" ></p>



## Multiagent RLlib: A unified official code releasement of MARL researches made by TJU-RL-Lab

This repository contains representative research works of TJU-RL-Lab on the topic of Multiagent Reinforcement Learning.

This repository will be constantly updated to include new researches made by TJU-RL-Lab.  

(The development of this repo is in progress at present.)



## Key Features

### :rocket: Including the State-Of-The-Art

- **[API-QMIX](https://arxiv.org/pdf/2203.05285.pdf)**: the state-of-the-art MARL algorithms in the [StarCraft Multi-Agent Challenge (SMAC)](https://github.com/oxwhirl/smac) benchmark. 
  - [SMAC](https://github.com/oxwhirl/smac) is [WhiRL](http://whirl.cs.ox.ac.uk/)'s environment for research in the field of collaborative multi-agent reinforcement learning (MARL) based on [Blizzard](http://blizzard.com/)'s [StarCraft II](https://en.wikipedia.org/wiki/StarCraft_II:_Wings_of_Liberty) RTS game. SMAC makes use of Blizzard's [StarCraft II Machine Learning API](https://github.com/Blizzard/s2client-proto) and [DeepMind](https://deepmind.com/)'s [PySC2](https://github.com/deepmind/pysc2) to provide a convenient interface for autonomous agents to interact with StarCraft II, getting observations and performing actions. SMAC concentrates on *decentralised micromanagement* scenarios, where each unit of the game is controlled by an individual RL agent.<img src="./assets/smac.webp" alt="SMAC" style="zoom:70%;" />



## Directory Structure (an overall view of research works in this repo)

| Category          | Sub-Categories                                   | Research Work (Conference) @Author | Progress |
| :-------------------------- | :---------------------------------------------------------- | :----------------------------------- | ------------------------------------ |
| **network_design**          | (1) action semantics; <br />(2) permutation invariant (equivariant) | (1) [ASN (ICLR-2020) @Weixun Wang](https://openreview.net/forum?id=ryg48p4tPH)<br />(2) [API (underreview) @Xiaotian Hao](https://arxiv.org/pdf/2203.05285.pdf) | :white_check_mark: |
| **credit_assignment**       |                                                             | (1) [QPD (ICML-2020) @Yaodong Yang](http://proceedings.mlr.press/v119/yang20d/yang20d.pdf)<br />(2) [Qatten (Arxiv) @Yaodong Yang](https://arxiv.org/abs/2002.03939) | :white_check_mark: |
| **multiagent_exploration**  |                                                             | [PMIC (NIPS-2021 workshop) @Pengyi Li](https://www.cooperativeai.com/neurips-2021/workshop-papers) | :no_entry:           |
| **large_scale_learning**    | (1) Game abstraction                                        | [G2ANet (AAAI-2020) @Weixun Wang](https://ojs.aaai.org/index.php/AAAI/article/view/6211) | :white_check_mark: |
| **curriculum_learning**     |                                                             | [DyAN (AAAI-2020) @Weixun Wang](https://ojs.aaai.org/index.php/AAAI/article/view/6221) | :white_check_mark:  |
| **hybrid_action**           |                                                             | [MAPQN/MAHHQN (IJCAI-2019) @Haotian Fu](MAPQN/MAHHQN (IJCAI-2019) @Haotian Fu) | :no_entry:                |
| **self_imitation_learning** |                                                             | [GASIL (AAMAS-2019) @Xiaotian Hao](https://www.ifaamas.org/Proceedings/aamas2019/pdfs/p1315.pdf) | :white_check_mark: |
| **non-stationarity**       |                              | (1) [BPR+ (NIPS-2018) @Yan Zheng](https://proceedings.neurips.cc/paper/2018/file/85422afb467e9456013a2a51d4dff702-Paper.pdf)<br />(2) [WDDQN  @Yan Zheng](https://arxiv.org/abs/1802.08534) <br />(3) [DPN-BPR+ (AAMAS2020) @Yan Zheng](https://link.springer.com/article/10.1007/s10458-020-09480-9) | :no_entry: |
| **hierarchical MARL**      |                                                        | [HIL/HCOMM/HQMIX (Arxiv) @Hongyao Tang](HIL/HCOMM/HQMIX (Arxiv) @Hongyao Tang) | :no_entry: |

