from typing import List


class RewardModel:
    def get_reward(self, current_state, action, next_state,):
        raise NotImplementedError

    def batch_rollout_reward(self, states, full_action_rollouts):
        raise NotImplementedError