from utils.configuration import DATASET_DIR
from utils.configuration import VIDEOS_CLIPS
from utils.video_tools import get_frames
import os
import cv2


out_dir = "/dice1-data/home/cabe0006/cvpr_experiments/trackformer_output/predictions/input"
# in_dir = "/dice1-data/home/cabe0006/cvpr_experiments/cvpr_data/dataset_v2_raw/videos"

for filename in VIDEOS_CLIPS:
    print('***********************************')
    print('***********************************')
    print('***********************************')
    print('***********************************')
    print('***********************************')
    print('***********************************')
    print(filename)
    frames = get_frames(os.path.join(DATASET_DIR, 'dataset_v2_raw', 'videos', f'{filename}.mp4'), max=720)
    image_dir = os.path.join(out_dir, filename)
    os.makedirs(image_dir, exist_ok=True)
    for i in range(540, 720):
        frame = frames[i]
        image_name = f'{i:06d}'
        cv2.imwrite(os.path.join(image_dir, f'{image_name}.jpg'), frame)



