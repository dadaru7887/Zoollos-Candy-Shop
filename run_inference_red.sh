#!/bin/bash
HF_USER=$(hf auth whoami | head -n 1 | sed 's/\x1b\[[0-9;]*m//g' | awk -F': ' '{print $2}' | xargs)

# eval 녹화는 매번 새로 덮어써도 되는 데이터라, 이전 결과 있으면 지우고 새로 시작
EVAL_DIR="$HOME/eval_single_red"
if [ -d "$EVAL_DIR" ]; then
  rm -rf "$EVAL_DIR"
fi

cd ~/il_ws/src/lerobot && lerobot-record \
  --robot.type=omx_follower \
  --robot.port=/dev/omx_follower \
  --robot.id=omx_follower_arm \
  --robot.cameras="{front: {type: opencv, index_or_path: '/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0', width: 640, height: 480, fps: 30, fourcc: 'MJPG'}, wrist: {type: opencv, index_or_path: '/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0', width: 640, height: 480, fps: 30, fourcc: 'MJPG'}}" \
  --display_data=true \
  --dataset.repo_id=${HF_USER}/eval_single_red \
  --dataset.root="$HOME/eval_single_red" \
  --dataset.single_task="Pick up Single Red" \
  --policy.path="$HOME/il_ws/src/lerobot/outputs/train/single_red_policy/checkpoints/last/pretrained_model" \
  --dataset.episode_time_s=100000 \
  --dataset.reset_time_s=1 \
  --dataset.push_to_hub=false
