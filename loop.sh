python main.py  \
--dataset_file ant2 \
--data_path /dice1-data/home/cabe0006/cvpr_experiments/cvpr_data/detection_source_dataset \
--data_path_target /dice1-data/home/cabe0006/cvpr_experiments/cvpr_data/detection_v2 \
--output_dir /dice1-data/home/cabe0006/cvpr_experiments/detr_output/new_detr_exp2 \
--batch_size 2 \
--resume /dice1-data/home/cabe0006/cvpr_experiments/detr_output/checkpoints_paper/detr-r101-2c7b67e5.pth \
--backbone resnet101 \
--name detr-target_newsmalldataset-source_prevsourcedataset \
--init \
--disc_loss_coef_local 100 \
--disc_loss_coef_global 100

