set -exo

export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_decompose_all_gather_einsum=true --xla_tpu_decompose_einsum_reduce_scatter=true --xla_tpu_scoped_vmem_limit_kib=98304 --xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_use_enhanced_launch_barrier=true --xla_tpu_enable_all_experimental_scheduler_features=true --xla_tpu_enable_scheduler_memory_pressure_tracking=true --xla_tpu_host_transfer_overlap_limit=2 --xla_tpu_aggressive_opt_barrier_removal=ENABLED --xla_lhs_prioritize_async_depth_over_stall=ENABLED --xla_tpu_enable_ag_backward_pipelining=true --xla_should_allow_loop_variant_parameter_in_chain=ENABLED --xla_should_add_loop_invariant_op_in_chain=ENABLED --xla_max_concurrent_host_send_recv=100 --xla_tpu_scheduler_percent_shared_memory_limit=100 --xla_latency_hiding_scheduler_rerun=2"

export PJRT_DEVICE=TPU
export XLA_USE_SPMD=1

mkdir -p profile
mkdir -p ir_dumps
mkdir -p xla_dumps
rm ir_dumps/scan-offload-ptxla.txt.* || true
rm -rf xla_dumps/scan-offload-ptxla || true
mkdir -p xla_dumps/scan-offload-ptxla

export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1
export PROFILE_EPOCH=0
export PROFILE_STEP=4
export PROFILE_DURATION_MS=120000
export PROFILE_LOGDIR=/workspaces/torch/profile/scan-offload-ptxla

mkdir -p /workspaces/torch/profile/scan-offload-ptxla

export XLA_SAVE_TENSORS_FILE=ir_dumps/scan-offload-ptxla.txt
export XLA_SAVE_TENSORS_FMT=hlo
export XLA_FLAGS=--xla_dump_to=xla_dumps/scan-offload-ptxla

# BLOCK_SIZE=4096
BLOCK_SIZE=2048
EXTRA='--flash_attention'

# Debugging notes:
# set print object on
# set print vtbl on
# b aten_xla_bridge.cpp:116
# set substitute-path torch_xla/csrc /workspaces/torch/pytorch/xla/torch_xla/csrc
# gdb --args python3 examples/pytorch/language-modeling/run_clm.py \
python3 examples/pytorch/language-modeling/run_clm.py \
  --dataset_name wikitext \
  --dataset_config_name wikitext-103-raw-v1 \
  --per_device_train_batch_size 8 \
  --do_train \
  --output_dir /workspaces/torch/output/test-clm \
  --overwrite_output_dir \
  --config_name /workspaces/torch/config.json \
  --cache_dir /workspaces/torch/cache \
  --tokenizer_name meta-llama/Llama-2-7b-hf \
  --block_size $BLOCK_SIZE \
  --optim adafactor \
  --save_strategy no \
  --logging_strategy no \
  --torch_dtype bfloat16 \
  --dataloader_drop_last yes \
  --spmd_2d_sharding 2 \
  --max_steps 500 $EXTRA
