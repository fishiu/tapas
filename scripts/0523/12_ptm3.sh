# finetune on ptm3 (no aug)
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

EXP=12_ptm3
echo "experiment:$EXP"

nohup /home/v-xiaojin/miniconda3/envs/table/bin/python tapas-finetune.py \
--shuffle \
--output_dir output/finetune/$EXP \
--model_name google/tapas-small \
--batch_size 32 \
--epochs 100 \
--pretrain_model output/pretrain/ptm3/checkpoints/100_36000_0.0740.pth \
> output/finetune/$EXP.nohup 2>&1 & echo $! > output/finetune/$EXP.pid