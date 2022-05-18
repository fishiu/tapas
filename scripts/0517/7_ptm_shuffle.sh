# run on a100
export PYTHONUNBUFFERED=1

EXP=7_ptm_shuffle
echo "experiment:$EXP"

nohup /anaconda/envs/table/bin/python tapas-finetune.py \
--shuffle \
--output_dir output/finetune/$EXP \
--model_name google/tapas-small \
--batch_size 32 \
--epochs 20 \
--pretrain_model output/pretrain/1_shuffle/checkpoints/49_12000_0.2868.pth\
> output/finetune/$EXP.nohup 2>&1 & echo $! > output/finetune/$EXP.pid