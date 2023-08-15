# RLGym-PPO
A vectorized implementation of PPO for use with [RLGym](https://github.com/lucas-emery/rocket-league-gym).

## INSTALLATION
`pip install rlgym-ppo`

## USAGE
Simply import the learner with `from rlgym_ppo import Learner`, pass it a function that will return an RLGym environment
and run the learning algorithm. A simple example follows:
```
from rlgym_ppo import Learner

def my_rlgym_function():
    import rlgym
    return rlgym.make()

learner = Learner(timestep_limit=50_000_000)
learner.learn(my_rlgym_env_function)
```
Note that users must implement a function to configure Rocket League (or RocketSim) in RLGym that returns an 
RLGym environment. See the `example.py` file for an example of writing such a function.
