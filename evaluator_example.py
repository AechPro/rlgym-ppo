TICK_SKIP = 8


def build_rlgym_v2_env():
    import numpy as np
    from rlgym.api import RLGym
    from rlgym.rocket_league import common_values
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import (
        GoalCondition,
        NoTouchTimeoutCondition,
    )
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.reward_functions import (
        CombinedReward,
        GoalReward,
        TouchReward,
    )
    from rlgym.rocket_league.sim import RLViserRenderer, RocketSimEngine
    from rlgym.rocket_league.state_mutators import (
        FixedTeamSizeMutator,
        KickoffMutator,
        MutatorSequence,
    )

    from rlgym_ppo.util import RLGymV2GymWrapper

    spawn_opponents = True
    team_size = 1
    blue_team_size = team_size
    orange_team_size = team_size if spawn_opponents else 0
    timeout_seconds = 10

    action_parser = RepeatAction(LookupTableAction(), repeats=TICK_SKIP)
    termination_condition = GoalCondition()
    truncation_condition = NoTouchTimeoutCondition(timeout=timeout_seconds)

    goal_reward_and_weight = (GoalReward(), 10)
    touch_reward_and_weight = (TouchReward(), 0.1)
    rewards_and_weights = (goal_reward_and_weight, touch_reward_and_weight)

    reward_fn = CombinedReward(*rewards_and_weights)

    obs_builder = DefaultObs(
        zero_padding=None,
        pos_coef=np.asarray(
            [
                1 / common_values.SIDE_WALL_X,
                1 / common_values.BACK_NET_Y,
                1 / common_values.CEILING_Z,
            ]
        ),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
    )

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=blue_team_size, orange_size=orange_team_size),
        KickoffMutator(),
    )
    rlgym_env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
        renderer=RLViserRenderer(),
    )

    return RLGymV2GymWrapper(rlgym_env)


if __name__ == "__main__":
    import os

    from rlgym_ppo import Evaluator

    # load the latest checkpoint
    checkpoint_load_folder = "data/checkpoints/"
    checkpoint_load_folder += (
        str(
            max(os.listdir(checkpoint_load_folder), key=lambda d: int(d.split("-")[-1]))
        )
        + "/"
    )
    checkpoint_load_folder += str(
        max(os.listdir(checkpoint_load_folder), key=lambda d: int(d))
    )

    evaluator = Evaluator(
        build_rlgym_v2_env,
        checkpoint_load_folder=checkpoint_load_folder,
        standardize_obs=False,
        render_delay=(120 / TICK_SKIP) / 120,
    )
    evaluator.eval()
