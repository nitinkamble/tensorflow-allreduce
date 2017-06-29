dir=/root/nitin/gpunode/tensorflow-allreduce.git

#srun --partition=K80x2 --ntasks=2 --gres=gpu:2 \
srun --ntasks=2 --gres=gpu:2 \
  python2.7 $dir/allreduce-test.py \
  --train-data train.txt \
  --validation-data train.txt \
  --vocab vocab.txt \
  --vocab-size 10000 \
  --batch-size 32 \
  --max-iterations 10000

