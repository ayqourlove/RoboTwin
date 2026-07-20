CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/train.py \
  pi05_base_adjust_bottle_singlearm_lora \
  --exp-name=adjust_bottle_singlearm_lora \
  --overwrite \
  --batch-size=16