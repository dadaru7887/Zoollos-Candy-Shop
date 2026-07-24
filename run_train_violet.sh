#!/bin/bash
HF_USER=$(hf auth whoami | head -n 1 | sed 's/\x1b\[[0-9;]*m//g' | awk -F': ' '{print $2}' | xargs)

cd ~/il_ws/src/lerobot && lerobot-train \
  --dataset.repo_id=${HF_USER}/dataset_single_violet_v2 \
  --policy.type=act \
  --policy.device=cuda \
  --policy.repo_id=${HF_USER}/single_violet_act_policy \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/single_violet_policy \
  --job_name=act_single_violet \
  --batch_size=8 \
  --steps=50000 \
  --log_freq=200 \
  --save_freq=1000 \
  --save_checkpoint=true
