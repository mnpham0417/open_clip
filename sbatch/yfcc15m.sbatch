#!/bin/bash

#SBATCH --output=yfcc15m_rn50_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --mem=192GB
#SBATCH --gres=gpu:4
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=yfcc15m_rn50
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=mp5847@nyu.edu

module purge

#env vars
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export MASTER_ADDR="$(hostname -s).hpc.nyu.edu"

singularity exec --nv \
    --overlay /scratch/work/public/imagenet/imagenet-train.sqf:ro \
	--overlay /scratch/work/public/imagenet/imagenet-val.sqf:ro \
    --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
    --overlay /vast/work/public/ml-datasets/yfcc15m/data/data-00.sqf:ro \
    --overlay /vast/work/public/ml-datasets/yfcc15m/data/data-01.sqf:ro \
    --overlay /vast/work/public/ml-datasets/yfcc15m/data/data-02.sqf:ro \
    --overlay /vast/work/public/ml-datasets/yfcc15m/data/data-03.sqf:ro \
    --overlay /vast/work/public/ml-datasets/yfcc15m/data/data-04.sqf:ro \
    --overlay /vast/work/public/ml-datasets/yfcc15m/data/data-05.sqf:ro \
    --overlay /vast/work/public/ml-datasets/yfcc15m/data/data-06.sqf:ro \
    --overlay /vast/work/public/ml-datasets/yfcc15m/data/data-07.sqf:ro \
    --overlay /vast/work/public/ml-datasets/yfcc15m/data/data-08.sqf:ro \
    --overlay /vast/work/public/ml-datasets/yfcc15m/data/data-09.sqf:ro \
    --overlay /vast/work/public/ml-datasets/yfcc15m/data/data-10.sqf:ro \
    --overlay /vast/work/public/ml-datasets/yfcc15m/data/data-11.sqf:ro \
    --overlay /vast/work/public/ml-datasets/yfcc15m/data/data-12.sqf:ro \
    --overlay /vast/work/public/ml-datasets/yfcc15m/data/data-13.sqf:ro \
    --overlay /vast/work/public/ml-datasets/yfcc15m/data/data-14.sqf:ro \
    --overlay /vast/work/public/ml-datasets/yfcc15m/data/data-15.sqf:ro \
    --overlay /vast/work/public/ml-datasets/yfcc15m/data/data-16.sqf:ro \
    --overlay /vast/work/public/ml-datasets/yfcc15m/data/data-17.sqf:ro \
    --overlay /vast/work/public/ml-datasets/yfcc15m/data/data-18.sqf:ro \
    --overlay /vast/work/public/ml-datasets/yfcc15m/data/data-19.sqf:ro \
    --overlay /vast/work/public/ml-datasets/yfcc15m/data/data-20.sqf:ro \
    --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
    --overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro \
	--overlay /scratch/mp5847/singularity_containers/overlay-50G-10M.ext3:ro \
	/scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
	/bin/bash -c "source /ext3/env.sh; conda activate /scratch/mp5847/src/comp_exp; pwd; cd src/; \
                    python3 -m training.main \
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
                    --workers=8 \
                    --model RN50 \
                    --report-to wandb \
                    --name 'CLIP RN50 yfcc15m' \
                    --csv-separator ',' \
                    --epochs=32 \
                    --local-loss \
                    --gather-with-grad "


