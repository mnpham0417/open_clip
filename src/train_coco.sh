#debug flags
echo $SLURM_JOB_NAME

#env vars
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export MASTER_ADDR="$(hostname -s).hpc.nyu.edu"

cd /home/mp5847/src/open_clip
export PYTHONPATH="$PYTHONPATH:$PWD/src"

python3 -m training.main \
            --save-frequency 1 \
            --zeroshot-frequency 1 \
            --train-data='/home/mp5847/src/open_clip/coco_caption_open_clip.csv'  \
            --csv-img-key image_id \
            --csv-caption-key caption \
            --imagenet-val=/imagenet/val \
            --warmup 2000 \
            --batch-size=256 \
            --lr=5e-4 \
            --wd=0.1 \
            --epochs=35 \
            --workers=4 \
            --model RN50 \
            --report-to wandb \
            --name 'CLIP RN50 COCO Caption' \
            --csv-separator ',' \
            --logs /scratch/mp5847/open-clip/logs/