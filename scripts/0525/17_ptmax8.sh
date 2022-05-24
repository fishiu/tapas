# run on a100
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

EXP=17_ptmax8
echo "experiment:$EXP"

nohup /anaconda/envs/table/bin/python tapas-finetune.py \
--shuffle \
--output_dir output/finetune/$EXP \
--model_name google/tapas-small \
--batch_size 32 \
--epochs 100 \
--pretrain_model output/pretrain/ptmax8/checkpoints/???????????.pth \
> output/finetune/$EXP.nohup 2>&1 & echo $! > output/finetune/$EXP.pid