# Deep Multi-Agent Reinforcement Learning with Discrete-Continuous Hybrid Action Spaces
Original PyTorch implementation of MAPQN and MAHHQN from Deep Multi-Agent Reinforcement Learning with Discrete-Continuous Hybrid Action Spaces

## Methods

**MAPQN** Deep Multi-Agent Parameterized Q-networks (Deep MAPQN), extends the architecture of P-DQN to multi-agent settings by leveraging the idea of Qmix

**MAHHQN**  Deep MAHHQN consists of a high-level network coordinating the learning over joint discrete actions and a low-level network for learning the coordinated policy over
the continuous parameters. We train the high-level network and low-level network separately and both of them follow the centralized training but decentralized execution paradigm.


# Installation
Known dependencies: Python (3.8.12), OpenAI gym (0.10.5), torch (1.8.1+cu102), numpy (1.19.5), Multi-agent Particle Environment, Half Feild Offense

## How to run
- `python3 src/main.py --config=<ALGORITHM> --env-config=<ENVIRONMENT> with <PARAMETERS>`:Run an ALGORITHM from the folder src/config/algs in an ENVIRONMENT from the folder src/config/envs on a specific GPU using some PARAMETERS
- `python3 src/main.py --config=iddpg_pp --env-config=particle with env_args.scenario_name=continuous_pred_prey_3a t_max=2000000` Run MAPQN from the folder src/config/algs in hybrid-MPE from the folder src/config/envs on a specific GPU using some PARAMETERS




## Citation

If you use our method or code in your research, please consider citing the paper as follows:

```
@article{fu2019deep,
  title={Deep multi-agent reinforcement learning with discrete-continuous hybrid action spaces},
  author={Fu, Haotian and Tang, Hongyao and Hao, Jianye and Lei, Zihan and Chen, Yingfeng and Fan, Changjie},
  journal={arXiv preprint arXiv:1903.04959},
  year={2019}
}
```


## License & Acknowledgements

MAPQN & MAHHQN is licensed under the MIT license. [MPE](https://github.com/openai/multiagent-particle-envs) are licensed under the Apache 2.0 license. 
