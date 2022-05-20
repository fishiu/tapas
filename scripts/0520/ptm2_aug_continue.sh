export PYTHONUNBUFFERED=1

EXP=2_aug_continue
echo "experiment:$EXP"

nohup /anaconda/envs/table/bin/python pretrain.py \
--train_json data/pretrain/totto/totto_merge_data.jsonl \
--output_dir=output/pretrain/$EXP \
--batch_size 200 \
--save_step 2000 \
--shuffle \
--aug w2v syno trans \
--load_checkpoint output/pretrain/2_aug/checkpoints/39_14000_0.1506 \
--start_total_step 14001 \
> output/pretrain/$EXP.out 2>&1 & \
echo $! > output/pretrain/$EXP.pid
