#debug flags
echo $SLURM_JOB_NAME

#env vars
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export MASTER_ADDR="$(hostname -s).hpc.nyu.edu"

cd /home/mp5847/src/open_clip
export PYTHONPATH="$PYTHONPATH:$PWD/src"

torchrun --nproc_per_node=4 -m training.main \
            --save-frequency 1 \
            --zeroshot-frequency 1 \
            --train-data='/vast/work/public/ml-datasets/yfcc15m/data/yfcc-small-metadata.csv'  \
            --csv-img-key filepath \
            --csv-caption-key title \
            --imagenet-val=/imagenet/val \
            --warmup 2000 \
            --batch-size=256 \
            --lr=1e-3 \
            --wd=0.1 \
            --epochs=30 \
            --workers=4 \
            --model RN50 \
            --report-to wandb \
            --name 'CLIP RN50 yfcc15m test' \
            --csv-separator ',' \
            --epochs=32