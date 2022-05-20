# more epoch
export PYTHONUNBUFFERED=1

EXP=10_ptm2
echo "experiment:$EXP"

nohup /anaconda/envs/table/bin/python tapas-finetune.py \
--shuffle \
--output_dir output/finetune/$EXP \
--model_name google/tapas-small \
--batch_size 32 \
--epochs 100 \
--pretrain_model output/pretrain/2_aug/checkpoints/39_14000_0.1506.pth \
> output/finetune/$EXP.nohup 2>&1 & echo $! > output/finetune/$EXP.pid