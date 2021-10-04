## Local Development Environment

python main.py --device cpu --num_workers 0 --dataset_file ant --data_path /Users/cabe0006/Projects/monash/Datasets/dataset-small/ --output_dir /Users/cabe0006/Projects/monash/detr/output/output_5 --resume /Users/cabe0006/Projects/monash/detr/checkpoints/detr-r50-e632da11.pth
python main.py --device cpu --num_workers 0 --dataset_file ant --data_path /Users/cabe0006/Projects/monash/Datasets/dataset-small/ --output_dir /Users/cabe0006/Projects/monash/detr/output/output_5 --resume /Users/cabe0006/Projects/monash/detr/output/output_5/checkpoint.pth --eval

python test.py --device cpu --dataset_file ant --data_path /Users/cabe0006/Projects/monash/Datasets/eval --output_dir /Users/cabe0006/Projects/monash/detr/output/output_3 --resume /Users/cabe0006/Projects/monash/detr/output/output_3/checkpoint.pth

python visualizer.py --device cpu --dataset_file ant --data_path /Users/cabe0006/Projects/monash/Datasets/eval --output_dir /Users/cabe0006/Projects/monash/Datasets/eval_output2 --resume /Users/cabe0006/Projects/monash/detr/output/output_4/checkpoint.pth



## Server Environment

# Start training
/home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --dataset_file ant --data_path /home/cabe0006/mb20_scratch/chamath/detr/dataset/ --output_dir /home/cabe0006/mb20_scratch/chamath/detr/output_backbone_freeze --resume /home/cabe0006/mb20_scratch/chamath/detr/output_backbone_freeze/checkpoint.pth
/home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --dataset_file ant --data_path /home/cabe0006/mb20_scratch/chamath/detr/dataset/ --output_dir /home/cabe0006/mb20_scratch/chamath/detr/output_backbone_freeze --eval --resume /home/cabe0006/mb20_scratch/chamath/detr/output_backbone_freeze/checkpoint.pth
# Train with Albumtation for image augmentations
/home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --dataset_file ant2 --data_path /home/cabe0006/mb20_scratch/chamath/detr/dataset/ --output_dir /home/cabe0006/mb20_scratch/chamath/detr/output_new_augmentations --resume /home/cabe0006/mb20_scratch/chamath/detr/output_new_augmentations/checkpoint.pth




# Resume training
/home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --dataset_file ant --data_path /home/cabe0006/mb20_scratch/chamath/detr/dataset/ --output_dir /home/cabe0006/mb20_scratch/chamath/detr/output --resume /home/cabe0006/mb20_scratch/chamath/detr/output/checkpoint.pth
# Draw box images
/home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env visualizer.py --dataset_file ant --data_path /home/cabe0006/mb20_scratch/chamath/detr-v2/dataset/test/ --output_dir /home/cabe0006/mb20_scratch/chamath/detr-v2/eval_output/test --resume /home/cabe0006/mb20_scratch/chamath/detr-v2/checkpoints/checkpoint.pth
/home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env visualizer.py --dataset_file ant --data_path /home/cabe0006/mb20_scratch/chamath/data/frames/sample31/ --output_dir /home/cabe0006/mb20_scratch/chamath/data/detr_eval/sample31 --resume /home/cabe0006/mb20_scratch/chamath/detr/output_new_augmentations/checkpoint.pth

/home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env visualizer.py --dataset_file ant --data_path /home/cabe0006/mb20_scratch/chamath/detr/dataset/eval/ --output_dir /home/cabe0006/mb20_scratch/chamath/detr/eval_output_with_new_augmentations2/eval --resume /home/cabe0006/mb20_scratch/chamath/detr/output_new_augmentations/checkpoint.pth

# Training with DC5
/home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env main_dc5.py --backbone resnet101 --dilation --dataset_file ant --data_path /home/cabe0006/mb20_scratch/chamath/detr/dataset/ --output_dir /home/cabe0006/mb20_scratch/chamath/detr/output_dc5 --resume /home/cabe0006/mb20_scratch/chamath/detr/checkpoint/detr-r101-dc5-a2e86def.pth


###################################NEW DATASET############################################

/home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --dataset_file ant2 --data_path /home/cabe0006/mb20_scratch/chamath/detr-v2/dataset/ --output_dir /home/cabe0006/mb20_scratch/chamath/detr-v2/checkpoints --resume /home/cabe0006/mb20_scratch/chamath/detr-v2/checkpoints/checkpoint.pth


/home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --dataset_file ant2 --data_path /home/cabe0006/mb20_scratch/chamath/detr-v3/dataset/ --output_dir /home/cabe0006/mb20_scratch/chamath/detr-v3/checkpoints --resume /home/cabe0006/mb20_scratch/chamath/detr-v3/checkpoints/checkpoint.pth
/home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env visualizer.py --dataset_file ant --data_path /home/cabe0006/mb20_scratch/chamath/data/ant_dataset_images_eval/untagged --output_dir /home/cabe0006/mb20_scratch/chamath/detr-v3/eval_output/test --resume /home/cabe0006/mb20_scratch/chamath/detr-v3/checkpoints/checkpoint.pth


/home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env visualizer.py --dataset_file ant --data_path /home/cabe0006/mb20_scratch/chamath/detr-v3/evaluation_results/images2 --output_dir /home/cabe0006/mb20_scratch/chamath/detr-v3/evaluation_results/videos --resume /home/cabe0006/mb20_scratch/chamath/detr-v3/checkpoints/checkpoint.pth


##DETR V4
/home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --dataset_file ant2 --data_path /home/cabe0006/mb20_scratch/chamath/detr-v4/dataset/ --output_dir /home/cabe0006/mb20_scratch/chamath/detr-v4/checkpoints --resume /home/cabe0006/mb20_scratch/chamath/detr-v4/checkpoints/checkpoint.pth
/home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env visualizer.py --dataset_file ant --data_path /home/cabe0006/mb20_scratch/chamath/data/ant_dataset_images_eval/untagged --output_dir /home/cabe0006/mb20_scratch/chamath/detr-v4/test_results --resume /home/cabe0006/mb20_scratch/chamath/detr-v4/checkpoints/checkpoint.pth
/home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env visualizer.py --dataset_file ant --data_path /home/cabe0006/mb20_scratch/chamath/detr-v3/raw_data_for_synthetic_dataset/synthetic_dataset/images --output_dir /home/cabe0006/mb20_scratch/chamath/detr-v4/artificial_test_results --resume /home/cabe0006/mb20_scratch/chamath/detr-v4/checkpoints/checkpoint.pth

/home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env visualizer.py --dataset_file ant --data_path /home/cabe0006/mb20_scratch/chamath/data/ant_dataset_images_eval/untagged --output_dir /home/cabe0006/mb20_scratch/chamath/detr-v4/test_results_1 --resume /home/cabe0006/mb20_scratch/chamath/detr-v4/checkpoints/checkpoint.pth

##DETR V5
/home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --dataset_file ant2 --data_path /home/cabe0006/mb20_scratch/chamath/detr-v5/dataset/ --output_dir /home/cabe0006/mb20_scratch/chamath/detr-v5/checkpoints --resume /home/cabe0006/mb20_scratch/chamath/detr-v5/checkpoints/checkpoint.pth
/home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env visualizer.py --dataset_file ant --data_path /home/cabe0006/mb20_scratch/chamath/data/ant_dataset_images_eval/untagged --output_dir /home/cabe0006/mb20_scratch/chamath/detr-v4/test_results --resume /home/cabe0006/mb20_scratch/chamath/detr-v4/checkpoints/checkpoint.pth
/home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env visualizer.py --dataset_file ant --data_path /home/cabe0006/mb20_scratch/chamath/detr-v3/raw_data_for_synthetic_dataset/synthetic_dataset/images --output_dir /home/cabe0006/mb20_scratch/chamath/detr-v5/artificial_test_results --resume /home/cabe0006/mb20_scratch/chamath/detr-v5/checkpoints/checkpoint.pth

/home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env visualizer.py --dataset_file ant --data_path /home/cabe0006/mb20_scratch/chamath/data/ant_dataset_images_eval/untagged --output_dir /home/cabe0006/mb20_scratch/chamath/detr-v5/test_results_1 --resume /home/cabe0006/mb20_scratch/chamath/detr-v5/checkpoints/checkpoint.pth


#v3 - with only originals
#v4 - with synthetic also
#v5 - with synthetic only


##Eval visualizations with v4
/home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env visualizer.py --dataset_file ant --data_path /home/cabe0006/mb20_scratch/chamath/data/evaluation_27/images_in --output_dir /home/cabe0006/mb20_scratch/chamath/data/evaluation_27/predictions/detr_v4_50 --resume /home/cabe0006/mb20_scratch/chamath/detr-v4/checkpoints/checkpoint.pth --thresh 0.5
/home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env visualizer.py --dataset_file ant --data_path /home/cabe0006/mb20_scratch/chamath/data/evaluation_27/images_in --output_dir /home/cabe0006/mb20_scratch/chamath/data/evaluation_27/predictions/detr_v4_75 --resume /home/cabe0006/mb20_scratch/chamath/detr-v4/checkpoints/checkpoint.pth --thresh 0.75


###################################################CVPR Trainings############################################################
/home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --dataset_file ant2 --data_path /home/cabe0006/mb20_scratch/chamath/cvpr_experiments/cvpr_data/detection_dataset --output_dir /home/cabe0006/mb20_scratch/chamath/cvpr_experiments/detr_output/checkpoints_kl_v2 --resume /home/cabe0006/mb20_scratch/chamath/cvpr_experiments/detr_output/checkpoints_kl_v2/checkpoint.pth --kl_div --checkpoint_freq 1 --batch_size 4
#/home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --dataset_file ant2 --data_path /home/cabe0006/mb20_scratch/chamath/cvpr_experiments/cvpr_data/detection_dataset --output_dir /home/cabe0006/mb20_scratch/chamath/cvpr_experiments/detr_output/checkpoints_kl_v2 --resume /home/cabe0006/mb20_scratch/chamath/cvpr_experiments/detr_output/checkpoints_kl_v1/checkpoint.pth --kl_div --checkpoint_freq 1

/home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env visualizer.py --dataset_file ant2 --data_path /home/cabe0006/mb20_scratch/chamath/cvpr_experiments/cvpr_data/detection_dataset/test --output_dir /home/cabe0006/mb20_scratch/chamath/cvpr_experiments/detr_output/temp_predictions/adapt_detr/kl_test --resume /home/cabe0006/mb20_scratch/chamath/cvpr_experiments/detr_output/checkpoints_kl_v2/checkpoint.pth

### MAP values for training dataset
/home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env train_evaluation.py --dataset_file ant2 --data_path /dice1-data/home/cabe0006/cvpr_experiments/cvpr_data/detection_dataset_small --output_dir /home/cabe0006/mb20_scratch/chamath/cvpr_experiments/detr_output/train_map_values_freeze_generator_2 --resume /dice1-data/home/cabe0006/cvpr_experiments/detr_output/checkpoints_freeze_generator_2/checkpoint0000.pth --batch_size 8 --lr_backbone 0.0001 --eval



#Local environment
python main.py --device cpu --num_workers 0 --dataset_file ant2 --data_path /Users/cabe0006/Projects/monash/cvpr_data/detection_dataset/local_env/ --output_dir /Users/cabe0006/Projects/monash/detr_v1/checkpoints --resume /Users/cabe0006/Projects/monash/detr_v1/checkpoints/detr-r50-e632da11.pth




