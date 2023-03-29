#!/bin/bash
#for i in CU10L1B1In_0 CU10L1B1Out_0 CU10L1B4In_0 CU10L1B4Out_0 CU10L1B5In_0 CU10L1B5Out_0 CU15L1B1In_0 CU15L1B1Out_0 CU15L1B4In_0 CU15L1B4Out_0 OU10B1L1In_0 OU10B1L1Out_0 OU10B1L2In_0 OU10B1L2Out_0 OU10B1L3In_0 OU10B1L3Out_0 OU10B2L1In_0 OU10B2L1Out_0 OU10B2L2In_0 OU10B2L2Out_0 OU10B2L3In_0 OU10B2L3Out_0 OU10B3L1In_0 OU10B3L1Out_0 OU10B3L2In_0 OU10B3L2Out_0 OU10B3L3In_0 OU10B3L3Out_0
#for i in CU10L1B6In_0 CU15L1B1In_0 CU15L1B4In_0 CU20L1B1Out_0 CU20L1B4Out_0 CU30L1B6Out_0
for i in task_switching_test
do
  /dice1-data/home/cabe0006/Projects/py_environments/env/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env visualizer.py \
  --dataset_file ant \
  --data_path /dice1-data/home/cabe0006/cvpr_experiments/trackformer_output/predictions/input/$i \
  --output_dir /dice1-data/home/cabe0006/cvpr_experiments/trackformer_output/predictions/detr_detections_test/target_$i \
  --resume /dice1-data/home/cabe0006/cvpr_experiments/detr_output/detr_task_switching_target/checkpoint0021.pth \
  --thresh 0.94

done