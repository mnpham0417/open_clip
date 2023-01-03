cd /home/mp5847/src/open_clip
export PYTHONPATH="$PYTHONPATH:$PWD/src"

python3 -m training.main \
            --save-frequency 1 \
            --zeroshot-frequency 0 \
            --train-data='/scratch/mp5847/open-clip/coco_caption_open_clip.csv'  \
            --csv-img-key image_id \
            --csv-caption-key caption \
            --warmup 2000 \
            --batch-size=128 \
            --lr=5e-4 \
            --wd=0.1 \
            --epochs=50 \
            --workers=6 \
            --model RN50 \
            --report-to wandb \
            --name 'CLIP RN50 COCO Caption TripletLoss(3 + 4) + ClipLoss - lambda=10' \
            --csv-separator ',' \
            --logs /scratch/mp5847/open-clip/logs  \
            --coco_instances_train_path '/coco/annotations/instances_train2017.json' \
            --coco_annotation_train_path '/coco/annotations/captions_train2017.json' \
            --coco_root_path '/coco/train2017' \
            --complement_categories_path '/scratch/mp5847/open-clip/complement_categories_dict.pkl'
