import wandb
import os
from tqdm import tqdm
from algorithms.common import Buffer
import torch
import torch.optim as optim
from typing import Tuple, List


class Logger:
    def log(self, data):
        raise NotImplementedError


class DefaultLogger(Logger):
    def log(self, data):
        print(data)


class WandbLogger(Logger):
    def __init__(self, hp_config):

        project = os.getenv("WANDB_PROJECT", "default-rl-project")
        entity = os.getenv("WANDB_ENTITY", None)

        wandb.init(
            project=project,
            entity=entity,
            config=hp_config.__dict__
        )

    def log(self, data):
        print(data)
        wandb.log(data)


class RLAlgorithm:
    def __init__(self, config, env, policy, evaluation_env):
        self.config = config
        self.env = env
        self.policy = policy
        self.gamma = config.gamma
        self.evaluation_env = evaluation_env

        if config.logger is None:
            self.logger = DefaultLogger()
        elif config.logger == "wandb":
            self.logger = WandbLogger(self.config)
        else:
            raise NotImplementedError("Logger not implemented")

    def stepwise_evaluate(self, n_eval_steps: int = 10, max_trajectory_steps: int = 10, **kwargs) -> float:
        total_rewards = 0

        for iteration in tqdm(range(n_eval_steps)):
            rewards = []
            states, actions, returns = [], [], []

            state = self.evaluation_env.reset()
            print("-" * 100)
            print("working on state: ", state)
            assert state.generated_solution == "", "restarted environment with non-empty generated solution"

            for step in range(max_trajectory_steps):
                action = self.policy.get_action(state)
                next_state, reward, done, _ = self.evaluation_env.step(action)

                # Store state, action, and reward
                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state  # Update the current state

                if done:
                    print("-" * 100)
                    print("Episode done:")
                    print(state)
                    print("-" * 100)
                    break

            total_rewards += sum(rewards)

        self.logger.log({"evaluation_reward": total_rewards})
        return total_rewards

    def direct_evaluate(self, n_eval_steps: int = 10, **kwargs) -> float:
        total_rewards = 0
        evaluate_batch_size = kwargs.get("evaluate_batch_size", self.config.evaluation_batch_size)

        for _ in tqdm(range(n_eval_steps)):
            states = [self.evaluation_env.reset() for _ in range(evaluate_batch_size)]
            full_action_rollouts = self.policy.batch_action_rollout(states)
            rewards = self.evaluation_env.reward_model.batch_rollout_reward(states, full_action_rollouts)
            total_rewards += sum(rewards)

        self.logger.log({"evaluation_reward": total_rewards})
        return total_rewards


    def evaluate(self, n_eval_steps: int = 10, max_trajectory_steps: int = 10, stepwise: bool = True, **kwargs) -> float:

        if self.evaluation_env is None:
            print("Evaluation environment not provided. Skipping evaluation.")
            return

        if stepwise:
            return self.stepwise_evaluate(n_eval_steps=n_eval_steps, max_trajectory_steps=max_trajectory_steps, **kwargs)
        else:
            print("Batch evaluation...")
            return self.direct_evaluate(n_eval_steps=n_eval_steps, **kwargs)

        
        
    def train(self, num_iterations=1, num_collection_episodes: int = 1, max_trajectory_steps: int = 10,
              num_update_steps: int = 2, load_buffer_from_cache_at_start=False, cache_buffer: bool = False,
              buffer_cache_path: str = None, n_eval_steps: int = 20, n_save_iterations: int = None,
              evaluate_at_start: bool = False, evaluation_iterations: int = 10, gradient_accumulation_steps: int = 16,
              save_path: str = "model.pt", stepwise_eval: bool = False, save_best_model: bool = False,
              update_ref_iterations=100, **kwargs
              ) -> None:

        scheduler = optim.lr_scheduler.LinearLR(
            self.policy.optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=num_iterations
        )

        max_trajectory_steps = getattr(self.config, "max_trajectory_steps", max_trajectory_steps)
        best_reward = evaluation_reward = float("-inf")

        assert gradient_accumulation_steps < num_update_steps, "gradient_accumulation_steps should be less than num_update_steps"

        if load_buffer_from_cache_at_start:
            assert buffer_cache_path is not None, "Buffer cache path is not provided."
            self.buffer = Buffer.load_from_disk(buffer_cache_path)
            print("Initialized buffer with {} examples".format(len(self.buffer.states)))

            # avoid overwriting the initial buffer by versioning the cache file
            if cache_buffer:
                buffer_cache_path = buffer_cache_path.replace(".pkl", "_updated.pkl")

        if evaluate_at_start and self.evaluation_env is not None:
            print("Initial evaluation: ")
            evaluation_reward = self.evaluate(max_trajectory_steps=max_trajectory_steps, n_eval_steps=n_eval_steps,
                                              stepwise=stepwise_eval)

        for iteration in range(num_iterations):
            total_rewards = 0
            self.logger.log({"lr": self.policy.get_current_lr()})

            # collecting episodes
            for episode in range(num_collection_episodes):

                rewards = []
                states, actions, returns = [], [], []

                state = self.env.reset()
                print("-"*100)
                print("working on state: ", state)
                assert state.generated_solution == "", "restarted environment with non-empty generated solution"

                # Generate an episode
                for step in range(max_trajectory_steps):
                    action = self.policy.get_action(state)
                    next_state, reward, done, _ = self.env.step(action)

                    # Store state, action, and reward
                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    total_rewards += reward

                    state = next_state  # Update the current state

                    if done:
                        break

                # Compute returns and update policy
                returns.extend(self.compute_returns(rewards))
                print(f"Episode {episode + 1}/{num_collection_episodes} completed.")

                # Add the episode to the buffer
                self.buffer.add(states, actions, returns)

                if cache_buffer:
                    assert buffer_cache_path is not None, "Buffer cache path is not provided."
                    self.buffer.save_to_disk(buffer_cache_path)

            average_reward = total_rewards / max(1.0, num_collection_episodes)
            print("Collection of episodes completed. Average reward: ", average_reward)
            self.logger.log({"average_reward": average_reward})

            # Update the policy
            # num_update_steps: int, iteration: int, num_iterations: int
            self.update_policy(num_update_steps, iteration, num_iterations, gradient_accumulation_steps,
                               update_ref_iterations=update_ref_iterations)

            if (iteration + 1) % evaluation_iterations == 0 and self.evaluation_env is not None:
                evaluation_reward = self.evaluate(max_trajectory_steps=max_trajectory_steps, n_eval_steps=n_eval_steps,
                                                  stepwise=stepwise_eval)

            if (n_save_iterations is not None) and (iteration + 1) % n_save_iterations == 0:
                if not save_best_model:
                    print("Saving model to: ", save_path)
                    self.policy.save_model_and_tokenizer(save_path)
                elif evaluation_reward > best_reward:
                    best_reward = evaluation_reward
                    print("Evaluation reward improved. Saving model to: ", save_path)
                    self.policy.save_model_and_tokenizer(save_path)

            scheduler.step()


    def compute_returns(self, rewards) -> List[float]:
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)  # Prepend to the list for correct order
        return returns

    def update_policy(self, num_update_steps: int, iteration: int, num_iterations: int,
                      gradient_accumulation_steps: int, update_ref_iterations: int, **kwargs) -> None:
        raise NotImplementedError
