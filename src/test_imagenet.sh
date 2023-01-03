cd /home/mp5847/src/open_clip
export PYTHONPATH="$PYTHONPATH:$PWD/src"

python3 -m training.main \
    --imagenet-val /imagenet/val \
    --model RN50 \
    --pretrained "/scratch/mp5847/open-clip/logs/CLIP RN50 COCO Caption TripletLoss(2 + 3 + 4) + ClipLoss - lambda=10/checkpoints/epoch_50.pt" \
    --batch-size 256 \
    --coco_instances_train_path '/coco/annotations/instances_train2017.json' \
    --coco_annotation_train_path '/coco/annotations/captions_train2017.json' \
    --coco_root_path '/coco/train2017' \
    --complement_categories_path '/scratch/mp5847/open-clip/complement_categories_dict.pkl'