# no aug
export PYTHONUNBUFFERED=1

EXP=ptmax8
echo "experiment:$EXP"

nohup /anaconda/envs/table/bin/python pretrain.py \
--dataset ar5iv \
--output_dir=output/pretrain/$EXP \
--batch_size 128 \
--save_step 2000 \
--shuffle \
--aug syno \
--aug_dir output/data/aug/ar5iv \
> output/pretrain/$EXP.out 2>&1 & \
echo $! > output/pretrain/$EXP.pid
