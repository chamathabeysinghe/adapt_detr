#!/bin/bash
for i in 52 53 54 55 56 57 58 59 60
do
  /home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env visualizer.py --dataset_file ant --data_path /home/cabe0006/mb20_scratch/chamath/data/ant_dataset_images/tagged/$i/ --output_dir /home/cabe0006/mb20_scratch/chamath/data/ant_dataset_detr_predictions/tagged/$i --resume /home/cabe0006/mb20_scratch/chamath/detr/output_new_augmentations/checkpoint.pth

#  python test.py --value=$i
#  python -c 'print "a"*'$i
done

