# no aug
export PYTHONUNBUFFERED=1

EXP=ptmax7
echo "experiment:$EXP"

nohup /anaconda/envs/table/bin/python pretrain.py \
--dataset ar5iv \
--output_dir=output/pretrain/$EXP \
--batch_size 128 \
--save_step 2000 \
--shuffle \
> output/pretrain/$EXP.out 2>&1 & \
echo $! > output/pretrain/$EXP.pid
