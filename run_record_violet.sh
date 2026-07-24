#!/bin/bash
HF_USER=$(hf auth whoami | head -n 1 | sed 's/\x1b\[[0-9;]*m//g' | awk -F': ' '{print $2}' | xargs)

RESUME_FLAG=false
if [ "$1" = "resume" ]; then
  RESUME_FLAG=true
fi

cd ~/il_ws/src/lerobot && lerobot-record \
  --robot.type=omx_follower \
  --robot.port=/dev/omx_follower \
  --robot.id=omx_follower_arm \
  --robot.cameras="{front: {type: opencv, index_or_path: '/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0', width: 640, height: 480, fps: 30, fourcc: 'MJPG'}, wrist: {type: opencv, index_or_path: '/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0', width: 640, height: 480, fps: 30, fourcc: 'MJPG'}}" \
  --teleop.type=omx_leader \
  --teleop.port=/dev/omx_leader \
  --teleop.id=omx_leader_arm \
  --display_data=true \
  --dataset.repo_id=${HF_USER}/dataset_single_violet \
  --dataset.single_task="Pick up Single Violet" \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=10 \
  --dataset.num_episodes=50 \
  --dataset.push_to_hub=false \
  --dataset.vcodec=h264_nvenc \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --resume=${RESUME_FLAG}
