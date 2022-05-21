# run on toy
export PYTHONUNBUFFERED=1

EXP=11_ptm2_continue
echo "experiment:$EXP"

nohup /home/v-xiaojin/miniconda3/envs/table/bin/python tapas-finetune.py \
--shuffle \
--output_dir output/finetune/$EXP \
--model_name google/tapas-small \
--batch_size 32 \
--epochs 100 \
--pretrain_model output/pretrain/2_aug_continue/checkpoints/100_50000_0.0627.pth \
> output/finetune/$EXP.nohup 2>&1 & echo $! > output/finetune/$EXP.pid