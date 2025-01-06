
from algorithms.rl_algorithm import RLAlgorithm
import torch
from algorithms.common import Buffer
from algorithms.reinfoce_config import ReinforceConfig
from typing import Tuple, List
from envs.env import Environment
from tqdm import tqdm

class Reinforce(RLAlgorithm):
    def __init__(self, policy, env, reward_model, config: ReinforceConfig, buffer: Buffer = None,
                 evaluation_env: Environment = None,
                 **kwargs):
        super(Reinforce, self).__init__(
            config=config,
            env=env,
            policy=policy,
            evaluation_env=evaluation_env
        )

        self.reward_model = reward_model

        if buffer is not None:
            self.buffer = buffer
        else:
            self.buffer = Buffer(config.buffer_size)

    def update_policy(self, num_update_steps: int, iteration: int, num_iterations: int,
                      gradient_accumulation_steps: int) -> None:

            total_loss = 0
            total_gradient_norm = 0
            self.policy.optimizer.zero_grad()

            for i in range(num_update_steps):
                states, actions, returns = self.buffer.sample(self.config.update_batch_size)

                returns = torch.tensor(returns, dtype=torch.float32, device=self.policy.device)
                assert not torch.isnan(returns).any(), "Returns contain NaN!"
                assert not torch.isinf(returns).any(), "Returns contain Inf!"
                returns = torch.clamp(returns, min=0.0, max=10.0)

                action_probabilities: Tuple[torch.Tensor, ...] = self.policy.calculate_action_probabilities(states, actions)
                assert not any(torch.isnan(prob).any() for prob in action_probabilities), "Action probabilities contain NaN!"
                logprobs = [torch.log(torch.clamp(prob, min=1e-10)).flatten() for prob in action_probabilities]


                policy_losses = []

                for logprob, G in zip(logprobs, returns):
                    # normalize by the number of tokens in the action sequence
                    policy_losses.append(-logprob * G / logprob.shape[0])

                loss = torch.cat(policy_losses).sum() / gradient_accumulation_steps
                loss.backward()

                if (i+1) % gradient_accumulation_steps == 0 or (i + 1) == num_update_steps:
                    total_gradient_norm += self.policy.calculate_average_gradient_norm()
                    self.policy.update(max_grad_norm=self.config.max_grad_norm)


                total_loss += loss.item()
                running_avg_loss = total_loss / (i+1 / gradient_accumulation_steps)

                if (i+1) % self.config.log_steps == 0 or (i + 1) == num_update_steps:
                    self.logger.log({"loss": running_avg_loss})
                    print("Current loss: ", running_avg_loss)


            avg_loss = total_loss / (num_update_steps / gradient_accumulation_steps)
            self.logger.log({"gradient_norm": total_gradient_norm / (num_update_steps / gradient_accumulation_steps)})
            print(f"Iteration {iteration + 1}/{num_iterations} completed. Loss: {avg_loss}")
