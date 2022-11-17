cd /home/mnpham/src/open_clip
export PYTHONPATH="$PYTHONPATH:$PWD/src"

python3 -m training.main \
            --save-frequency 1 \
            --zeroshot-frequency 0 \
            --train-data='/home/mnpham/Desktop/coco_caption_open_clip_local.csv'  \
            --csv-img-key image_id \
            --csv-caption-key caption \
            --warmup 2000 \
            --batch-size=5 \
            --lr=5e-4 \
            --wd=0.1 \
            --epochs=35 \
            --workers=4 \
            --model RN50 \
            --report-to wandb \
            --name 'CLIP RN50 COCO Caption Test' \
            --csv-separator ',' \
            --logs ./logs  \
            --coco_instances_train_path '/media/mnpham/HARD_DISK_3/dataset/coco_2017/annotations_trainval2017/annotations/instances_train2017.json' \
            --coco_annotation_train_path '/media/mnpham/HARD_DISK_3/dataset/coco_2017/annotations_trainval2017/annotations/captions_train2017.json' \
            --coco_root_path '/media/mnpham/HARD_DISK_3/dataset/coco_2017/train2017' \
            --complement_categories_path '/home/mnpham/Desktop/complement_categories_dict.pkl'
