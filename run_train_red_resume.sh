#!/bin/bash
cd ~/il_ws/src/lerobot && lerobot-train \
  --config_path=outputs/train/single_red_policy/checkpoints/last/pretrained_model/train_config.json \
  --resume=true \
  --save_freq=3000
