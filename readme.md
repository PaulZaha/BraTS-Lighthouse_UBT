Current readme for EfficientVit-SAM model.
Relevant folder is the src_EfficientViT


Virtual Environment

Download the BraTS Train and Validation Data and put the folders into src_EfficientViT/data/raw_data_BraTS

download the LiteMedSAM weights and put them in work_dir/LiteMedSAM/

download the EfficientViT-SAM l0 checkpoint from https://github.com/mit-han-lab/efficientvit/blob/master/applications/efficientvit_sam/README.md
rename it to efficientvit_sam_l0_checkpoint.pt and put into src_EfficientViT Folder


Data Strucruing:
python structure_BraTS_data.py

python create_edge_imgs.py

python create_dataset.py


Run for raw data:
python distill.py -num_epochs 70 -batch_size 7 -device cuda -work_dir work_dir_distill/ -resume work_dir_distill/medsam_lite_latest.pth -pretrained_checkpoint efficientvit_sam_l0_checkpoint.pt --traincsv data/dataset_csvs/train.csv --valcsv data/dataset_csvs/val.csv

python modelmerge.py work_dir_distill/medsam_lite_best.pth distilled.pth

python finetune.py -pretrained_checkpoint distilled.pth -num_epochs 70 -batch_size 1 -device cuda -work_dir work_dir_general -resume work_dir_general/medsam_lite_latest.pth --traincsv data/dataset_csvs/train.csv --valcsv data/dataset_csvs/val.csv




Run for kirsch data:
python distill.py -num_epochs 70 -batch_size 7 -device cuda -work_dir work_dir_distill_kirsch/ -resume work_dir_distill_kirsch/medsam_lite_latest.pth -pretrained_checkpoint efficientvit_sam_l0_checkpoint.pt --traincsv data/dataset_csvs/train_kirsch.csv --valcsv data/dataset_csvs/val_kirsch.csv

python modelmerge.py work_dir_distill/medsam_lite_best.pth distilled_kirsch.pth

python finetune.py -pretrained_checkpoint distilled_kirsch.pth -num_epochs 70 -batch_size 1 -device cuda -work_dir work_dir_general_kirsch -resume work_dir_general_kirsch/medsam_lite_latest.pth --traincsv data/dataset_csvs/train_kirsch.csv --valcsv data/dataset_csvs/val_kirsch.csv