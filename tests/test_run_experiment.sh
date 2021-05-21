#!/bin/bash
FAILURE=false

python run_experiment.py \
  --data_class=FakeData \
  --image_width=128 \
  --image_height=32 \
  --num_samples=128 \
  --conv_dim=16 \
  --model_class=CRNN \
  --max_epochs=2 \
  --max_lr=0.01 \
|| FAILURE=true


if [ "$FAILURE" = true ]; then
  echo "Test for run_experiment.py failed"
  exit 1
fi
echo "Test for run_experiment.py passed"
exit 0