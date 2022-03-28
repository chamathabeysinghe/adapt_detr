/home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py  \
				--dataset_file ant2 \
				--data_path /home/cabe0006/mb20_scratch/chamath/cvpr_experiments/cvpr_data/detection_source_dataset_small \
				--data_path_target /home/cabe0006/mb20_scratch/chamath/cvpr_experiments/cvpr_data/detection_target_dataset_small \
				--output_dir /home/cabe0006/mb20_scratch/chamath/cvpr_experiments/detr_output/detr_disc_exp5_small_dataset_coef_100_exp2weights \
				--batch_size 2 \
				--resume /home/cabe0006/mb20_scratch/chamath/cvpr_experiments/detr_output/detr_disc_exp5_small_dataset_coef_100_exp2weights/checkpoint.pth \
				--backbone resnet101 \
				--name detr_disc_exp5_small_dataset_coef_100_exp2weights \
				--disc_loss_coef 100
