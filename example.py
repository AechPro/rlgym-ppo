def build_rocketsim_env():
    import rlgym_sim
    from rlgym_sim.utils.reward_functions import CombinedReward
    from rlgym_sim.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityBallToGoalReward, \
        EventReward
    from rlgym_sim.utils.obs_builders import DefaultObs
    from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
    from rlgym_sim.utils import common_values
    from rlgym_sim.utils.action_parsers import  ContinuousAction
    import numpy as np

    spawn_opponents = True
    team_size = 1
    game_tick_rate = 120
    tick_skip = 8
    timeout_seconds = 10
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

    action_parser = ContinuousAction()
    terminal_conditions = [NoTouchTimeoutCondition(timeout_ticks), GoalScoredCondition()]

    rewards_to_combine = (VelocityPlayerToBallReward(),
                          VelocityBallToGoalReward(),
                          EventReward(team_goal=1, concede=1, shot=0.05, save=0.3, demo=0.1))
    reward_weights = (0.001, 0.01, 1.0)

    reward_fn = CombinedReward(reward_functions=rewards_to_combine,
                               reward_weights=reward_weights)

    obs_builder = DefaultObs(pos_coef=np.asarray([1/common_values.SIDE_WALL_X, 1/common_values.BACK_NET_Y, 1/common_values.CEILING_Z]),
                             ang_coef=1/np.pi,
                             lin_vel_coef=1/common_values.CAR_MAX_SPEED,
                             ang_vel_coef=1/common_values.CAR_MAX_ANG_VEL)

    env = rlgym_sim.make(tick_skip=tick_skip,
                         team_size=team_size,
                         spawn_opponents=spawn_opponents,
                         terminal_conditions=terminal_conditions,
                         reward_fn=reward_fn,
                         obs_builder=obs_builder,
                         action_parser=action_parser)

    return env

if __name__ == "__main__":
    from rlgym_ppo import Learner
    learner = Learner(build_rocketsim_env,
                      log_to_wandb=True)
    learner.learn()