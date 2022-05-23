# no aug version
export PYTHONUNBUFFERED=1

EXP=ptm4
echo "experiment:$EXP"

nohup /anaconda/envs/table/bin/python pretrain.py \
--train_json data/pretrain/totto/totto_merge_data.jsonl \
--output_dir=output/pretrain/$EXP \
--batch_size 128 \
--save_step 2000 \
--shuffle \
--aug syno \
> output/pretrain/$EXP.out 2>&1 & \
echo $! > output/pretrain/$EXP.pid
