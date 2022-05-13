export PYTHONUNBUFFERED=1
nohup /anaconda/envs/table/bin/python tapas-finetune.py --shuffle --output_dir=output/0508/2_epoch --epochs 40 > output/0508/2_epoch.nohup 2>&1 &