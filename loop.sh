python main.py  \
--dataset_file ant2 \
--data_path /dice1-data/home/cabe0006/cvpr_experiments/cvpr_data/detection_target_dataset \
--data_path_target /dice1-data/home/cabe0006/cvpr_experiments/cvpr_data/task_switching_dataset \
--output_dir /dice1-data/home/cabe0006/cvpr_experiments/detr_output/detr_task_switching_target \
--batch_size 2 \
--resume /dice1-data/home/cabe0006/cvpr_experiments/detr_output/detr_exp25/checkpoint_best_ap_50.pth \
--backbone resnet101 \
--name detr-task-switching-exp-target \
--init \
--disc_loss_coef_local 100 \
--disc_loss_coef_global 100 \
--checkpoint_freq 1




#--resume /dice1-data/home/cabe0006/cvpr_experiments/detr_output/checkpoints_paper/detr-r101-2c7b67e5.pth \
