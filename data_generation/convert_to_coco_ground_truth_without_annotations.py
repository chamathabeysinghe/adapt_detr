from utils.configuration import DATASET_DIR
from utils.configuration import VIDEOS_CLIPS
from utils.video_tools import get_frames
import pandas as pd
import json
import os
import cv2
import random

SCALE = 1.0
SKIP_INTERVAL = 1
DATASET_NAME = 'task_switching_dataset'
# split = 'train'
# file_names = ['task_switching_train', 'task_switching_val', 'task_switching_test']


COCO_DIR = os.path.join(DATASET_DIR, DATASET_NAME)
os.makedirs(COCO_DIR, exist_ok=True)

for split in ['train', 'val', 'test']:
    image_count = 0
    json_obj = {
        "categories": [
            {
                "id": 0,
                "name": "ant"
            }
        ],
        "images": [],
        "annotations": []
    }
    file = f'task_switching_{split}'
    frames = get_frames(os.path.join(DATASET_DIR, 'raw_data_task_switching', f'{file}.mp4'))
    image_dir = os.path.join(DATASET_DIR, DATASET_NAME, split)
    os.makedirs(image_dir, exist_ok=True)

    for image_id in range(len(frames)):
        # if image_id % SKIP_INTERVAL != 0:
        #     continue
        image_count += 1
        image_name = f'{file}_{image_id:06d}'
        cv2.imwrite(os.path.join(image_dir, f'{image_name}.jpg'), frames[image_id])
        json_obj["images"].append({
            "id": image_count,
            "license": 1,
            "file_name": "{}.jpg".format(image_name),
            "height": int(542.0 / SCALE),
            "width": int(1024.0 / SCALE),
            "date_captured": "null"
        })
    with open(os.path.join(COCO_DIR, f'ground-truth-{split}.json'), 'w') as outfile:
        json.dump(json_obj, outfile)




