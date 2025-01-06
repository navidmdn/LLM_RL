from algorithms.rl_algorithm import RLAlgorithm
import torch
from algorithms.common import Buffer
from algorithms.reinfoce_stable_config import ReinforceStableConfig
from typing import Tuple, List
from tqdm import tqdm
from envs.env import Environment


class ReinforceStable(RLAlgorithm):
    def __init__(self, policy, ref_policy, env, reward_model, config: ReinforceStableConfig, buffer: Buffer = None,
                 evaluation_env: Environment = None,
                 **kwargs):

        super(ReinforceStable, self).__init__(
            config=config,
            env=env,
            policy=policy,
            evaluation_env=evaluation_env
        )
        self.ref_policy = ref_policy
        self.reward_model = reward_model

        if buffer is not None:
            self.buffer = buffer
        else:
            self.buffer = Buffer(config.buffer_size)


    def update_policy(self, num_update_steps: int, iteration: int, num_iterations: int,
                      gradient_accumulation_steps: int, update_ref_iterations: int,  **kwargs) -> None:

        total_loss = 0
        total_gradient_norm = 0
        self.policy.optimizer.zero_grad()

        adaptive_update_steps = min(num_update_steps, self.buffer.current_size//self.config.update_batch_size)

        for i in tqdm(range(adaptive_update_steps)):
            states, actions, returns = self.buffer.sample(self.config.update_batch_size)

            returns = torch.tensor(returns, dtype=torch.float32, device=self.policy.device)
            assert not torch.isnan(returns).any(), "Returns contain NaN!"
            assert not torch.isinf(returns).any(), "Returns contain Inf!"
            returns = torch.clamp(returns, min=0.0, max=10.0)

            action_probabilities: Tuple[torch.Tensor, ...] = self.policy.calculate_action_probabilities(states, actions)
            ref_action_probabilities: Tuple[torch.Tensor, ...] = self.ref_policy.calculate_action_probabilities(
                states, actions, inference_mode=True)

            assert not any(
                torch.isnan(prob).any() for prob in action_probabilities), "Action probabilities contain NaN!"
            logprobs = [torch.log(torch.clamp(prob, min=1e-10)).flatten() for prob in action_probabilities]
            ref_logprobs = [torch.log(torch.clamp(prob, min=1e-10)).flatten().detach() for prob in
                            ref_action_probabilities]

            policy_losses = []
            divergences = []

            for logprob, ref_logprob, G in zip(logprobs, ref_logprobs, returns):
                # normalize by the number of tokens in the action sequence
                normalized_logprob = logprob / logprob.shape[0]
                normalized_ref_logprob = ref_logprob / ref_logprob.shape[0]
                divergences.append(torch.sum(normalized_logprob - normalized_ref_logprob).item())

                KL_return = G - self.config.kl_coeff * (normalized_logprob - normalized_ref_logprob)
                policy_losses.append(-normalized_logprob * KL_return)

            loss = torch.cat(policy_losses).sum() / gradient_accumulation_steps
            loss.backward()

            if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == num_update_steps:
                total_gradient_norm += self.policy.calculate_average_gradient_norm()
                self.policy.update(max_grad_norm=self.config.max_grad_norm)

            total_loss += loss.item()
            running_avg_loss = total_loss / (i + 1 / gradient_accumulation_steps)

            if (i + 1) % self.config.log_steps == 0 or (i + 1) == num_update_steps:
                self.logger.log({"loss": running_avg_loss})
                self.logger.log({"divergence": sum(divergences) / len(divergences)})
                print("Current loss: ", running_avg_loss)

        avg_loss = total_loss / (num_update_steps / gradient_accumulation_steps)
        self.logger.log(
            {"gradient_norm": total_gradient_norm / (num_update_steps / gradient_accumulation_steps)})
        print(f"Iteration {iteration + 1}/{num_iterations} completed. Loss: {avg_loss}")

        if (iteration + 1) % update_ref_iterations == 0:
            self.update_ref_policy_with_current()


    def update_ref_policy_with_current(self):
        self.ref_policy.model.load_state_dict(self.policy.model.state_dict())
        print("Reference policy updated.")
