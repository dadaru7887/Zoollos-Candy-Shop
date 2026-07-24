#!/bin/bash
HF_USER=$(hf auth whoami | head -n 1 | sed 's/\x1b\[[0-9;]*m//g' | awk -F': ' '{print $2}' | xargs)

cd ~/il_ws/src/lerobot && lerobot-train \
  --dataset.repo_id=${HF_USER}/grasp-candy \
  --policy.type=act \
  --policy.device=cuda \
  --policy.repo_id=${HF_USER}/grasp_candy_act_policy \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/grasp_candy_policy \
  --job_name=act_grasp-candy \
  --batch_size=8 \
  --steps=20000 \
  --save_freq=5000
