#WANDB_MODE=online WANDB_ENTITY=navidmdn WANDB_PROJECT=gsm8k-rl python sft_trainer.py\
#  --train_file ../data/train_all.json\
#  --dev_dir ../data/\
#  --output_dir outputs/llama3.2-1b-sft\
#  --do_train\
#  --do_eval\
#  --cache_dir ../../../hfcache\
#  --model_id meta-llama/Llama-3.2-1B-Instruct\
#  --per_device_train_batch_size 4\
#  --per_device_eval_batch_size 4\
#  --gradient_accumulation_steps 16\
#  --eval_accumulation_steps 1\
#  --torch_empty_cache_steps 1\
#  --batch_eval_metrics\
#  --num_train_epochs 1\
#  --save_strategy steps\
#  --eval_strategy steps\
#  --save_total_limit 3\
#  --metric_for_best_model "eval_val_accuracy"\
#  --include_for_metrics "inputs"\
#  --logging_steps 1\
#  --eval_on_start\
#  --overwrite_output_dir\
#  --report_to wandb\
#  --load_best_model_at_end\
#  --eval_steps 2\
#  --save_steps 2\
#  --max_new_tokens 1024\
#  --learning_rate 1.0e-5\
#  --lr_scheduler_type cosine\
#  --warmup_ratio 0.1\
#  --bf16\
#  --flash_attention\
#  --system_prompt_path ../policy/stepwise_solver_sys.txt

WANDB_MODE=online WANDB_ENTITY=navidmdn WANDB_PROJECT=gsm8k-rl python sft_trainer.py\
  --train_file ../data/train_all.json\
  --dev_dir ../data/\
  --output_dir outputs/llama3.2-3b-sft\
  --do_train\
  --do_eval\
  --cache_dir ../../../hfcache\
  --model_id meta-llama/Llama-3.2-3B-Instruct\
  --per_device_train_batch_size 2\
  --per_device_eval_batch_size 4\
  --gradient_accumulation_steps 32\
  --eval_accumulation_steps 1\
  --torch_empty_cache_steps 1\
  --batch_eval_metrics\
  --num_train_epochs 1\
  --save_strategy steps\
  --eval_strategy steps\
  --save_total_limit 1\
  --metric_for_best_model "eval_val_accuracy"\
  --include_for_metrics "inputs"\
  --logging_steps 1\
  --eval_on_start\
  --overwrite_output_dir\
  --report_to wandb\
  --load_best_model_at_end\
  --eval_steps 2\
  --save_steps 2\
  --max_new_tokens 1024\
  --learning_rate 1.0e-5\
  --lr_scheduler_type cosine\
  --warmup_ratio 0.1\
  --bf16\
  --flash_attention\
  --system_prompt_path ../policy/stepwise_solver_sys.txt
