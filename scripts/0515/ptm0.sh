export PYTHONUNBUFFERED=1

EXP=0_demo
echo "experiment:$EXP"

nohup /anaconda/envs/table/bin/python pretrain.py \
--output_dir=output/pretrain/$EXP \
--save_step 500 \
--batch_size 320 \
> output/pretrain/$EXP.out 2>&1 & \
echo $! > output/pretrain/$EXP.pid