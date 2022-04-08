# The curse of dimensionality (scalability) issue
In MARL, the joint state-action space grows exponentially as the number of agents increases. This is also referred to as the combinatorial nature of MARL. Thus, MARL algorithms typically suffer from **poor sample-efficiency** and **poor scalability** due to the exponential grows of the dimensionality. The key to solve this problem is to reduce the size of the state-action space properly. 

| Category        | Sub-Categories                                               | Research Work (Conference)                                   | Progress           |
| :-------------- | :----------------------------------------------------------- | :----------------------------------------------------------- | ------------------ |
| **scalability** | **scalable multiagent network**<br />    (1) permutation invariant (equivariant)   <br />    (2) action semantics<br />    (3) Game abstraction <br />    (4) dynamic agent-number network<br /><br />**hierarchical MARL**<br /> | (1) [API (underreview) [1]](https://arxiv.org/pdf/2203.05285.pdf) <br />(2) [ASN (ICLR-2020) [2]](https://openreview.net/forum?id=ryg48p4tPH)<br />(3) [G2ANet (AAAI-2020) [3]](https://ojs.aaai.org/index.php/AAAI/article/view/6211) <br />(4) [DyAN (AAAI-2020) [4]](https://ojs.aaai.org/index.php/AAAI/article/view/6221)<br />(5) [HIL/HCOMM/HQMIX (Arxiv) [5]](https://arxiv.org/pdf/1809.09332.pdf?ref=https://githubhelp.com) | :white_check_mark: |







## Publication List

[1] Hao X, Wang W, Mao H, et al. API: Boosting Multi-Agent Reinforcement Learning via Agent-Permutation-Invariant Networks[J]. arXiv preprint arXiv:2203.05285, 2022.

[2] Wang W, Yang T, Liu Y, et al. Action Semantics Network: Considering the Effects of Actions in Multiagent Systems[C]//International Conference on Learning Representations. 2019.

[3] Liu Y, Wang W, Hu Y, et al. Multi-agent game abstraction via graph attention neural network[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2020, 34(05): 7211-7218.

[4] Wang W, Yang T, Liu Y, et al. From few to more: Large-scale dynamic multiagent curriculum learning[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2020, 34(05): 7293-7300.

[5] Tang H, Hao J, Lv T, et al. Hierarchical deep multiagent reinforcement learning with temporal abstraction[J]. arXiv preprint arXiv:1809.09332, 2018.
