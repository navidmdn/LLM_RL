# LLM_RL
A repository to experiment with RL algorithms on LLMs

## Overview and Goal

The initial goal of creating this repo was to build a simple gym-like software to interact with a language 
environment (like MATH and GSM8K) and train a LLM using reinforcement learning to solve it.

My main concern was to have the simple RL training loop and an abstract paradigm to define different algorithms easily. This simple code base, along with a lot of flaws tries to mitigate this problem. At the time of writing this code, I hadn't found any other libs to do the same thing. trl repo from huggingface was the only option but I couldn't get the abstraction that I wanted.

## Navigating the code

You can start from `examples/reinforce_stable_on_gsm8k_training.py` to see how the training loop is defined. The `examples` folder contains a few examples to get you started.
You can define various RL algorithms in the `algorithms` folder. The `envs` folder contains the environment definitions.
I've also defined a `few_shot_llm_stepwise_math_policy` in the `policies` folder. This is a simple policy that uses a few-shot learning paradigm and a LLM as the underlying policy.

The parent `RLAlgorithm` class in `algorithms/rl_algorithm.py` defines the basic structure of an RL algorithm. You can define your own algorithm by inheriting this class and implementing the `train` method. This will simply lead you
to define a classic RL training loop as follows:

```python
def train(self):
    ...
    for episode in range(num_episodes):
        state = self.env.reset()
        for step in range(max_steps):
            action = self.policy.get_action(state)
            next_state, reward, done, _ = self.env.step(action)
            self.buffer.add(state, action, reward)
            self.update_policy()
```

A sample Reinforce algorithm is implemented in `algorithms/reinforce_stable.py`. You can define your own algorithm in a similar way.

## Experimenting with the code

You can run the following command to run the REINFORCE algorithm on the GSM8K environment:

```bash
WANDB_MODE=online WANDB_PROJECT="wandb-prj-name" WANDB_ENTITY=your-entity-name PYTHONPATH=. python examples/reinforce_stable_on_gsm8k_training.py\
 --gsm8k_path data/test_all.json\
 --train_data_file data/train_all.json\
 --test_gsm8k_path data/test_all.json\
 --sys_prompt_path policy/stepwise_solver_sys.txt\
 --model_name_or_path meta-llama/Llama-3.2-1B-Instruct\
 --ref_model_name meta-llama/Llama-3.2-1B-Instruct\
 --cache_dir ../../hfcache\
 --n_few_shot_examples 5\
 --num_iterations 20\
 --num_collection_episodes 100\
 --max_trajectory_steps 10\
 --num_update_steps 4096\
 --cache_buffer\
 --evaluation_iterations 1\
 --n_eval_steps 101\
 --buffer_cache_path data/buffer_cache_l1b_k25.pkl\
 --gradient_accumulation_steps 32\
 --save_path data/policy_l1b_k25\
 --n_save_iterations 1\
 --kl_coeff 0.25\
 --sym_calc\
 --evaluation_at_start\
 --update_ref_iterations 2
```

My experiment shows that a stable version of Reinforce algorithm (with a few modifications like keeping a reference model and calculating
KL divergence between the policy and the reference model) can enhance the `llama-3.2-1B-instruct` model's performance on gsm8k from 22% to 38% accuracy: ![img](/statics/reward_average.png)
