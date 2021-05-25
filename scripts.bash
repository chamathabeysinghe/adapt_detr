## Local Development Environment

python main.py --device cpu --num_workers 0 --dataset_file ant --data_path /Users/cabe0006/Projects/monash/Datasets/dataset-small/ --output_dir /Users/cabe0006/Projects/monash/detr/output/output_3 --resume /Users/cabe0006/Projects/monash/detr/checkpoints/detr-r50-e632da11.pth

python test.py --device cpu --dataset_file ant --data_path /Users/cabe0006/Projects/monash/Datasets/eval --output_dir /Users/cabe0006/Projects/monash/detr/output/output_3 --resume /Users/cabe0006/Projects/monash/detr/output/output_3/checkpoint.pth

python visualizer.py --device cpu --dataset_file ant --data_path /Users/cabe0006/Projects/monash/Datasets/eval --output_dir /Users/cabe0006/Projects/monash/Datasets/eval_output --resume /Users/cabe0006/Projects/monash/detr/output/output_3/checkpoint.pth



## Server Environment

# Start training
/home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --dataset_file ant --data_path /home/cabe0006/mb20_scratch/chamath/detr/dataset/ --output_dir /home/cabe0006/mb20_scratch/chamath/detr/output --resume /home/cabe0006/mb20_scratch/chamath/detr/checkpoint/detr-r50-e632da11.pth
# Resume training
/home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --dataset_file ant --data_path /home/cabe0006/mb20_scratch/chamath/detr/dataset/ --output_dir /home/cabe0006/mb20_scratch/chamath/detr/output --resume /home/cabe0006/mb20_scratch/chamath/detr/output/checkpoint.pth
# Draw box images
/home/cabe0006/mb20_scratch/chamath/detr/venv_detr/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env visualizer.py --dataset_file ant --data_path /home/cabe0006/mb20_scratch/chamath/detr/dataset/test/ --output_dir /home/cabe0006/mb20_scratch/chamath/detr/eval_output/test --resume /home/cabe0006/mb20_scratch/chamath/detr/output/checkpoint.pth
