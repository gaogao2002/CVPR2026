accelerate launch --gpu_ids 0 --mixed_precision "no" \
  train.py \
    --device "cuda" \
    --mixed_precision "no" \
    --gradient_checkpointing \
    \
    --count 8 \
    --num_train_epochs 50 \
    --early_stopping_patience 5 \
    --early_stopping_min_delta 0.0 \
    \
    --data_root_path "/root/autodl-tmp/datasets/FSC147-good" \
    --keep_density_acc 0  \
    --batch_size 1 \
    --gradient_accumulation_steps 1 \
    \
    --output_path "/root/autodl-tmp/results/density_control" \
    --stable_diffusion_pipeline_path "/root/autodl-tmp/stable-diffusion/sdxl-turbo" \
    \
    --lr 6e-4 \
    --betas1 0.9 \
    --betas2 0.999 \
    --weight_decay 0 \
    --eps 1e-8 \
    --max_grad_norm 1.0 \
    --seed 35 \
    --lambda_1 10 \
    --lambda_2 0.1 \
    \
    --save_steps 500 \
    \
    --guidance_scale 2.1 \
    --height 384 \
    --width 384 \
    \
    --placeholder_token "newcls" \
    --initializer_token "some" \
    \
    --hyperparam "param-0"
