#!/bin/bash
FAILURE=false

python run_experiment.py \
  --data_class=FakeData \
  --model_class=ResnetTransformer \
  --resnet_num_layers=3 \
  --tf_dim=128 \
  --tf_fc_dim=256 \
  --tf_num_layers=1 \
  --max_output_length=30 \
  --tf_nhead=1 \
  --max_epochs=2 \
|| FAILURE=true


#python run_experiment.py \
#  --data_class=FakeData \
#  --image_width=128 \
#  --image_height=32 \
#  --num_samples=128 \
#  --conv_dim=16 \
#  --model_class=CRNN \
#  --max_epochs=2 \
#  --max_lr=0.01 \
#|| FAILURE=true


if [ "$FAILURE" = true ]; then
  echo "Test for run_experiment.py failed"
  exit 1
fi
echo "Test for run_experiment.py passed"
exit 0