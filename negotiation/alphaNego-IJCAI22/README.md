# alpha-Nego
alpha-Nego: Self-play Deep Reinforcement Learning for Negotiation based on Natural Language
Implements our alpha-Nego algorithms.
A more clear code with proper instruction will be available soon here. 

----------
## Installation

### Dependencies

1. Create a conda environment:

python==3.5.6, pytorch==1.1.0

```shell
conda create --name <env> --file conda_requirements.txt
```

2. pip install additional dependencies
```shell
# install other dependecies
pip install -r pip_requirements.txt
# install our code
pip install -e .
```


## Traning instruction

### Preprocess
```shell
PYTHONPATH=. python core/price_tracker.py --train-examples-path data/train-luis-clean2.json --output data/price_tracker.pkl
```

### Baselines

### Train a Supervised Learning Agent
```shell
bash craigslistbargain/exp_scripts/identifier/old/train_sl.sh
```
### Train A2C Agent
Generate scenarios
```shell
PYTHONPATH=. python ../scripts/chat_to_scenarios.py --chats data/train-luis-post.json --scenarios data/train-scenarios.json
PYTHONPATH=. python ../scripts/chat_to_scenarios.py --chats data/dev-luis-post.json --scenarios data/dev-scenarios.json
```
Train the RL model 
```shell
bash exp_scripts/rl/train_a2c.sh
```
### Train ToM model
Sample data
```shell
bash exp_scripts/identifier/sample_data.sh
```
Implicit Model
```shell
bash exp_scripts/identifier/train_uttr_history_tom.sh
```
Explicit Model
```shell
bash exp_scripts/identifier/train_uttr_id_history_tom.sh
```
### Train alpha-Nego
```shell
bash exp_scripts/rl/train_nego.sh
```

### Evaluate result of different model.

Rule Model
```shell
bash exp_scripts/rl/eval_rule.sh
```
SL Model
```shell
bash exp_scripts/rl/eval_sl.sh
```
A2C Model
```shell
bash exp_scripts/rl/eval_rl.sh
```  
Implicit ToM Model
```shell
bash exp_scripts/rl/eval_tom_noid.sh
```
Explicit ToM Model
```shell
bash exp_scripts/rl/eval_tom.sh
```

alpha-Nego Model
```shell
bash exp_scripts/rl/eval_nego.sh
```  
