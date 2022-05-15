# run on toy
export PYTHONUNBUFFERED=1

EXP=3_small
echo "experiment:$EXP"

nohup /home/v-xiaojin/miniconda3/envs/table/bin/python tapas-finetune.py --shuffle \
--output_dir output/finetune/$EXP \
--model_name google/tapas-small \
--batch_size 64 \
--epochs 20 \
> output/finetune/$EXP.nohup 2>&1 & echo $! > output/finetune/$EXP.pid