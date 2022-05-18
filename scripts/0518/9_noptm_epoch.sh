# more epoch
export PYTHONUNBUFFERED=1

EXP=9_noptm_epoch
echo "experiment:$EXP"

nohup /anaconda/envs/table/bin/python tapas-finetune.py \
--shuffle \
--output_dir output/finetune/$EXP \
--model_name google/tapas-small \
--batch_size 32 \
--epochs 100 \
> output/finetune/$EXP.nohup 2>&1 & echo $! > output/finetune/$EXP.pid