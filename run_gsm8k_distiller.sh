
WANDB_MODE=online WANDB_PROJECT="gsm8k-distiller_rl" WANDB_ENTITY=navidmdn PYTHONPATH=. python examples/reinforce_stable_on_gsm8k_training_distillation.py\
 --gsm8k_path data/train_all.json\
 --train_data_file data/train_all.json\
 --test_gsm8k_path data/val.json\
 --sys_prompt_path policy/stepwise_solver_sys.txt\
 --model_name_or_path comparison_baselines/outputs/llama3.2-1b-sft\
 --ref_model_name meta-llama/Llama-3.2-3B-Instruct\
 --cache_dir ../../hfcache\
 --num_iterations 10\
 --num_collection_episodes 200\
 --max_trajectory_steps 10\
 --num_update_steps 4096\
 --cache_buffer\
 --evaluation_iterations 1\
 --n_eval_steps 101\
 --buffer_cache_path data/buffer_cache_l1b_distill_k100.pkl\
 --gradient_accumulation_steps 32\
 --save_path data/policy_l1b_distill_k100\
 --n_save_iterations 1\
 --nll_coeff 2.0\
 --sym_calc\
 --stepwise_eval