#
## added final step reward + start buffer + sym_calc
#WANDB_MODE=online WANDB_PROJECT="gsm8k-rl" WANDB_ENTITY=navidmdn PYTHONPATH=. python examples/teach_llm_with_llm_reinforce_gsm8k_multidim_reward.py\
# --gsm8k_path data/train_all.json\
# --test_gsm8k_path data/test_all.json\
# --sys_prompt_path policy/stepwise_solver_sys.txt\
# --model_name_or_path meta-llama/Llama-3.2-1B-Instruct\
# --reward_model_name meta-llama/Llama-3.1-8B-Instruct\
# --cache_dir ../../hfcache\
# --reward_model_sys_prompt_path envs/multi_dimensional_rewards.json\
# --num_iterations 20\
# --num_collection_episodes 100\
# --max_trajectory_steps 10\
# --num_update_steps 1024\
# --cache_buffer\
# --buffer_cache_path data/buffer_cache_llmWllm.pkl\
# --gradient_accumulation_steps 32\
# --save_path data/policy_llmWllm\
# --evaluation_iterations 2\
# --evaluation_at_start\
# --load_buffer_from_cache_at_start\
# --n_save_iterations 2\
# --n_eval_steps 100\
# --sym_calc\
# --kl_coeff 0.25\
# --gamma 0.9
#
#WANDB_MODE=online WANDB_PROJECT="gsm8k-rl" WANDB_ENTITY=navidmdn PYTHONPATH=. python examples/teach_llm_with_llm_reinforce_gsm8k_multidim_reward.py\
# --gsm8k_path data/train_all.json\
# --test_gsm8k_path data/test_all.json\
# --sys_prompt_path policy/stepwise_solver_sys.txt\
# --model_name_or_path meta-llama/Llama-3.2-3B-Instruct\
# --reward_model_name meta-llama/Llama-3.1-8B-Instruct\
# --cache_dir ../../hfcache\
# --reward_model_sys_prompt_path envs/multi_dimensional_rewards.json\
# --num_iterations 20\
# --num_collection_episodes 100\
# --max_trajectory_steps 10\
# --num_update_steps 1024\
# --cache_buffer\
# --load_buffer_from_cache_at_start\
# --buffer_cache_path data/buffer_cache_llmWllm.pkl\
# --gradient_accumulation_steps 32\
# --save_path data/policy_llmWllm\
# --evaluation_iterations 2\
# --evaluation_at_start\
# --n_save_iterations 2\
# --n_eval_steps 100\
# --sym_calc\
# --kl_coeff 0.75\
# --gamma 0.9

#WANDB_MODE=online WANDB_PROJECT="gsm8k-rl" WANDB_ENTITY=navidmdn PYTHONPATH=. python examples/teach_llm_with_llm_reinforce_gsm8k_multidim_reward.py\
# --gsm8k_path data/train_all.json\
# --test_gsm8k_path data/test_all.json\
# --train_data_file data/train_all.json\
# --sys_prompt_path policy/stepwise_solver_sys.txt\
# --model_name_or_path meta-llama/Llama-3.2-1B-Instruct\
# --reward_model_name meta-llama/Llama-3.1-70B-Instruct\
# --cache_dir ../../hfcache\
# --n_few_shot_examples 8\
# --reward_model_sys_prompt_path envs/multi_dimensional_rewards.json\
# --num_iterations 10\
# --num_collection_episodes 100\
# --max_trajectory_steps 10\
# --num_update_steps 1024\
# --cache_buffer\
# --buffer_cache_path data/buffer_cachea_llmWllm_1b_r70b.pkl\
# --gradient_accumulation_steps 32\
# --save_path data/policy_llmWllm_1b_r70b\
# --evaluation_iterations 1\
# --n_save_iterations 1\
# --n_eval_steps 82\
# --kl_coeff 0.25\
# --save_best_model\
# --quantize_reward


#WANDB_MODE=online WANDB_PROJECT="gsm8k-rl" WANDB_ENTITY=navidmdn PYTHONPATH=. python examples/reinforce_stable_on_gsm8k_training.py\
# --gsm8k_path data/train_all.json\
# --train_data_file data/train_all.json\
# --test_gsm8k_path data/test_all.json\
# --sys_prompt_path policy/stepwise_solver_sys.txt\
# --model_name_or_path meta-llama/Llama-3.2-1B-Instruct\
# --cache_dir ../../hfcache\
# --num_iterations 20\
# --num_collection_episodes 500\
# --max_trajectory_steps 10\
# --num_update_steps 4096\
# --cache_buffer\
# --evaluation_iterations 2\
# --n_eval_steps 83\
# --buffer_cache_path data/buffer_cache_l1b_kl50_updateref.pkl\
# --gradient_accumulation_steps 32\
# --save_path data/policy_l3b_kl50_updateref\
# --n_save_iterations 2\
# --kl_coeff 0.5\
# --sym_calc\
# --update_ref_iterations 1

WANDB_MODE=online WANDB_PROJECT="gsm8k-rl" WANDB_ENTITY=navidmdn PYTHONPATH=. python examples/reinforce_stable_on_gsm8k_training_distillation.py\
 --gsm8k_path data/train_all.json\
 --train_data_file data/train_all.json\
 --test_gsm8k_path data/val.json\
 --sys_prompt_path policy/stepwise_solver_sys.txt\
 --model_name_or_path comparison_baselines/outputs/llama3.2-1b-sft\
 --ref_model_name meta-llama/Llama-3.2-3B-Instruct\
 --cache_dir ../../hfcache\
 --num_iterations 20\
 --num_collection_episodes 100\
 --max_trajectory_steps 10\
 --num_update_steps 4096\
 --cache_buffer\
 --evaluation_iterations 1\
 --n_eval_steps 101\
 --buffer_cache_path data/buffer_cache_l1b_distill_k100.pkl\
 --gradient_accumulation_steps 32\
 --save_path data/policy_l1b_distill_k100\
 --n_save_iterations 1\
 --kl_coeff 1.0\
 --sym_calc\
 --stepwise_eval