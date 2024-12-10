#!/bin/bash

pip install .

python3 examples/pytorch/language-modeling/run_clm.py \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --per_device_train_batch_size 32 \
  --do_train \
  --output_dir /tmp/test-clm \
  --overwrite_output_dir \
  --config_name mixtral_config.json \
  --cache_dir /tmp \
  --tokenizer_name mistralai/Mixtral-8x7B-v0.1 \
  --block_size 4096 \
  --optim adafactor \
  --save_strategy no \
  --logging_strategy no \
  --fsdp "full_shard" \
  --fsdp_config fsdp_config.json \
  --torch_dtype bfloat16 \
  --dataloader_drop_last yes \
  --flash_attention \
  --num_train_epochs 1 \
  --static
