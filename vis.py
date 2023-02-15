import cv2
import os
import numpy as np


def write_file(vid_frames, file_name):
    height, width, layers = vid_frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(file_name, fourcc, 6.0, (width, height))

    for i in range(len(vid_frames)):
        out.write((vid_frames[i]).astype(np.uint8))
    out.release()


in_dir = "/dice1-data/home/cabe0006/cvpr_experiments/trackformer_output/predictions/detr_detections"

dirs = ['colony3_small_day_2', 'colony5_small_day_2', 'colony6_small_day_1', 'colony7_small_day_1']

for file in dirs:
    print(file)
    d = os.path.join(in_dir, file)
    imgs = [cv2.imread(os.path.join(d, f)) for f in sorted(os.listdir(d))]
    write_file(imgs, d+'.mp4')

