# run on toy
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1

EXP=15_ptm6
echo "experiment:$EXP"

nohup /home/v-xiaojin/miniconda3/envs/table/bin/python tapas-finetune.py \
--shuffle \
--output_dir output/finetune/$EXP \
--model_name google/tapas-small \
--batch_size 32 \
--epochs 100 \
--pretrain_model output/pretrain/ptm4/checkpoints/100_56000_0.0471.pth \
> output/finetune/$EXP.nohup 2>&1 & echo $! > output/finetune/$EXP.pid