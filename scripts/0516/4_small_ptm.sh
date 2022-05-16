# run on a100
export PYTHONUNBUFFERED=1

EXP=4_small_ptm
echo "experiment:$EXP"

nohup /anaconda/envs/table/bin/python tapas-finetune.py --shuffle \
--output_dir output/finetune/$EXP \
--model_name google/tapas-small \
--batch_size 512 \
--epochs 20 \
--pretrain_model output/pretrain/0_demo/checkpoints/61_15000_0.1684.pth\
> output/finetune/$EXP.nohup 2>&1 & echo $! > output/finetune/$EXP.pid