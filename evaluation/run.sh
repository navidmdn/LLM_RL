#PYTHONPATH=.. python generate_gsm8k_responses.py\
# --gsm8k_path ../data/test_all.json\
# --sys_prompt_path ../policy/stepwise_solver_sys.txt\
# --model_name_or_path meta-llama/Llama-3.1-8B-Instruct\
# --cache_dir ../../../hfcache\
# --reward_model_sys_prompt_path ../envs/reward_prompt.txt\
# --output_path ../data/gsm8k_test_responses.json
#
#PYTHONPATH=.. python generate_gsm8k_responses.py\
# --gsm8k_path ../data/test_all.json\
# --sys_prompt_path ../policy/stepwise_solver_sys.txt\
# --model_name_or_path meta-llama/Llama-3.1-8B-Instruct\
# --cache_dir ../../../hfcache\
# --reward_model_sys_prompt_path ../envs/reward_prompt.txt\
# --output_path ../data/gsm8k_test_responses_sym.json\
# --sym_calc

PYTHONPATH=.. python generate_gsm8k_responses.py\
 --gsm8k_path ../data/test_all.json\
 --sys_prompt_path ../policy/stepwise_solver_sys.txt\
 --model_name_or_path ../comparison_baselines/outputs/llama3.2-1b-sft\
 --cache_dir ../../../hfcache\
 --output_path ../data/gsm8k_test_sft_sym.json\
 --sym_calc