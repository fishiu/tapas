# run on a100
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

EXP=16_ptmax7
echo "experiment:$EXP"

nohup /anaconda/envs/table/bin/python tapas-finetune.py \
--shuffle \
--output_dir output/finetune/$EXP \
--model_name google/tapas-small \
--batch_size 32 \
--epochs 100 \
--pretrain_model output/pretrain/ptmax7/checkpoints/99_18000_0.1742.pth \
> output/finetune/$EXP.nohup 2>&1 & echo $! > output/finetune/$EXP.pid