export PYTHONPATH="${PYTHONPATH}:/home/v-xiaojin/repos/table/code/tapas"
for i in {1..10} ; do
  echo "Running preproc.sh for subject $i"
  nohup /anaconda/envs/table/bin/python pipeline.py --html_dir ../data/ar5iv/$i --csv_dir ../data/ar5iv_csv/$i --task_id $i > ../output/0510/preproc/$i.nohup 2>&1 &
done
# /anaconda/envs/table/bin/python pipeline.py --html_dir ../data/ar5iv/10 --csv_dir ../data/ar5iv_csv/10 --task_id 10

# rm output/0510/*.nohup
# rm -r data/ar5iv_csv/*