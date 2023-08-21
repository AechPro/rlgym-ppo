# RLGym-PPO
A vectorized implementation of PPO for use with [RLGym](rlgym.org).

## INSTALLATION
1. install [RLGym-sim](https://github.com/AechPro/rocket-league-gym-sim). 
2. If you would like to use a GPU install [PyTorch with CUDA](https://pytorch.org/get-started/locally/)
3. Install this project via `pip install git+https://github.com/AechPro/rlgym-ppo`

## USAGE
Simply import the learner with `from rlgym_ppo import Learner`, pass it a function that will return an RLGym environment
and run the learning algorithm. A simple example follows:
```
from rlgym_ppo import Learner

def my_rlgym_function():
    import rlgym_sim
    return rlgym_sim.make()

learner = Learner(my_rlgym_env_function)
learner.learn()
```
Note that users must implement a function to configure Rocket League (or RocketSim) in RLGym that returns an 
RLGym environment. See the `example.py` file for an example of writing such a function.
