export PYTHONUNBUFFERED=1
nohup /anaconda/envs/table/bin/python pretrain.py --output_dir=output/pretrain/0_demo --batch_size 320 > output/pretrain/0_demo/nohup.out 2>&1 &