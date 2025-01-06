
#WANDB_MODE=online WANDB_PROJECT="gsm8k-rl" WANDB_ENTITY=navidmdn PYTHONPATH=. python examples/reinforce_stable_on_gsm8k_training.py\
# --gsm8k_path data/train_all.json\
# --train_data_file data/train_all.json\
# --test_gsm8k_path data/val.json\
# --sys_prompt_path policy/stepwise_solver_sys.txt\
# --model_name_or_path comparison_baselines/outputs/llama3.2-1b-sft\
# --ref_model_name meta-llama/Llama-3.2-1B-Instruct\
# --cache_dir ../../hfcache\
# --num_iterations 20\
# --num_collection_episodes 100\
# --max_trajectory_steps 10\
# --num_update_steps 4096\
# --cache_buffer\
# --evaluation_iterations 1\
# --n_eval_steps 101\
# --buffer_cache_path data/buffer_cache_l1b_k50.pkl\
# --gradient_accumulation_steps 32\
# --save_path data/policy_l1b_k50\
# --n_save_iterations 1\
# --kl_coeff 0.5\
# --sym_calc\
# --stepwise_eval\
# --evaluation_at_start\
# --update_ref_iterations 1

#WANDB_MODE=online WANDB_PROJECT="gsm8k-rl-GHexp" WANDB_ENTITY=navidmdn PYTHONPATH=. python examples/reinforce_stable_on_gsm8k_training.py\
# --gsm8k_path data/test_all.json\
# --train_data_file data/train_all.json\
# --test_gsm8k_path data/test_all.json\
# --sys_prompt_path policy/stepwise_solver_sys.txt\
# --model_name_or_path comparison_baselines/outputs/llama3.2-1b-sft\
# --ref_model_name comparison_baselines/outputs/llama3.2-1b-sft\
# --cache_dir ../../hfcache\
# --n_few_shot_examples 0\
# --num_iterations 20\
# --num_collection_episodes 100\
# --max_trajectory_steps 10\
# --num_update_steps 4096\
# --cache_buffer\
# --evaluation_iterations 1\
# --n_eval_steps 101\
# --buffer_cache_path data/buffer_cache_l1b_k25.pkl\
# --gradient_accumulation_steps 32\
# --save_path data/policy_l1b_k25\
# --n_save_iterations 1\
# --kl_coeff 0.25\
# --sym_calc\
# --evaluation_at_start\
# --update_ref_iterations 2

#WANDB_MODE=online WANDB_PROJECT="gsm8k-rl-GHexp" WANDB_ENTITY=navidmdn PYTHONPATH=. python examples/reinforce_stable_on_gsm8k_training.py\
# --gsm8k_path data/test_all.json\
# --train_data_file data/train_all.json\
# --test_gsm8k_path data/test_all.json\
# --sys_prompt_path policy/stepwise_solver_sys.txt\
# --model_name_or_path meta-llama/Llama-3.2-1B-Instruct\
# --ref_model_name meta-llama/Llama-3.2-1B-Instruct\
# --cache_dir ../../hfcache\
# --n_few_shot_examples 5\
# --num_iterations 20\
# --num_collection_episodes 100\
# --max_trajectory_steps 10\
# --num_update_steps 4096\
# --cache_buffer\
# --evaluation_iterations 1\
# --n_eval_steps 101\
# --buffer_cache_path data/buffer_cache_l1b_k25_5shot.pkl\
# --gradient_accumulation_steps 32\
# --save_path data/policy_l1b_k25_5shot\
# --n_save_iterations 1\
# --kl_coeff 0.25\
# --sym_calc\
# --evaluation_at_start\
# --update_ref_iterations 2

WANDB_MODE=online WANDB_PROJECT="gsm8k-rl-GHexp" WANDB_ENTITY=navidmdn PYTHONPATH=. python examples/reinforce_stable_on_gsm8k_training.py\
 --gsm8k_path data/test_all.json\
 --train_data_file data/train_all.json\
 --test_gsm8k_path data/test_all.json\
 --sys_prompt_path policy/stepwise_solver_sys.txt\
 --model_name_or_path comparison_baselines/outputs/llama3.2-3b-sft\
 --ref_model_name comparison_baselines/outputs/llama3.2-3b-sft\
 --cache_dir ../../hfcache\
 --n_few_shot_examples 0\
 --num_iterations 20\
 --num_collection_episodes 100\
 --max_trajectory_steps 10\
 --num_update_steps 4096\
 --cache_buffer\
 --evaluation_iterations 1\
 --n_eval_steps 101\
 --buffer_cache_path data/buffer_cache_l3b_k25.pkl\
 --gradient_accumulation_steps 32\
 --save_path data/policy_l3b_k25\
 --n_save_iterations 1\
 --kl_coeff 0.25\
 --sym_calc\
 --evaluation_at_start\
 --update_ref_iterations 2