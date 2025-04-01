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

python modalities3D.py data/dataset_csvs/train.csv data/dataset_csvs/val.csv data/dataset_csvs/train_kirsch.csv data/dataset_csvs/val_kirsch.csv



Run for raw data:
python distill.py -num_epochs 70 -batch_size 7 -device cuda -work_dir work_dir_distill/ -resume work_dir_distill/medsam_lite_latest.pth -pretrained_checkpoint efficientvit_sam_l0_checkpoint.pt --traincsv data/dataset_csvs/train.csv --valcsv data/dataset_csvs/val.csv

python modelmerge.py work_dir_distill/medsam_lite_best.pth distilled.pth

python finetune.py -pretrained_checkpoint distilled.pth -num_epochs 70 -batch_size 1 -device cuda -work_dir work_dir_general -resume work_dir_general/medsam_lite_latest.pth --traincsv data/dataset_csvs/train.csv --valcsv data/dataset_csvs/val.csv

python extract_evit.py work_dir_general/medsam_lite_best.pth general_finetuned.pth

python daft.py -pretrained_checkpoint general_finetuned.pth -num_epochs 70 -batch_size 1 -device cuda -work_dir work_dir_modalities3D/CT/ -resume work_dir_modalities3D/CT/medsam_lite_latest.pth --traincsv data/dataset_csvs/modalities3D/CT.train.csv --valcsv data/dataset_csvs/modalities3D/CT.val.csv
python daft.py -pretrained_checkpoint general_finetuned.pth -num_epochs 70 -batch_size 1 -device cuda -work_dir work_dir_modalities3D/BraTS/ -resume work_dir_modalities3D/BraTS/medsam_lite_latest.pth --traincsv data/dataset_csvs/modalities3D/BraTS.train.csv --valcsv data/dataset_csvs/modalities3D/BraTS.val.csv

mkdir models && ./extract_modalities3D.sh
cp general_finetuned.pth models/general.pth

Run for kirsch data:
python distill.py -num_epochs 70 -batch_size 7 -device cuda -work_dir work_dir_distill_kirsch/ -resume work_dir_distill_kirsch/medsam_lite_latest.pth -pretrained_checkpoint efficientvit_sam_l0_checkpoint.pt --traincsv data/dataset_csvs/train_kirsch.csv --valcsv data/dataset_csvs/val_kirsch.csv

python modelmerge.py work_dir_distill/medsam_lite_best.pth distilled_kirsch.pth

python finetune.py -pretrained_checkpoint distilled_kirsch.pth -num_epochs 70 -batch_size 1 -device cuda -work_dir work_dir_general_kirsch -resume work_dir_general_kirsch/medsam_lite_latest.pth --traincsv data/dataset_csvs/train_kirsch.csv --valcsv data/dataset_csvs/val_kirsch.csv

python extract_evit.py work_dir_general_kirsch/medsam_lite_best.pth general_finetuned_kirsch.pth

python daft.py -pretrained_checkpoint general_finetuned_kirsch.pth -num_epochs 70 -batch_size 1 -device cuda -work_dir work_dir_modalities3D_kirsch/CT/ -resume work_dir_modalities3D_kirsch/CT/medsam_lite_latest.pth --traincsv data/dataset_csvs/modalities3D_kirsch/CT_kirsch.train.csv --valcsv data/dataset_csvs/modalities3D_kirsch/CT_kirsch.val.csv
python daft.py -pretrained_checkpoint general_finetuned_kirsch.pth -num_epochs 70 -batch_size 1 -device cuda -work_dir work_dir_modalities3D_kirsch/BraTS/ -resume work_dir_modalities3D_kirsch/BraTS/medsam_lite_latest.pth --traincsv data/dataset_csvs/modalities3D_kirsch/BraTS_kirsch.train.csv --valcsv data/dataset_csvs/modalities3D_kirsch/BraTS_kirsch.val.csv

mkdir models_kirsch && ./extract_modalities3D_kirsch.sh
cp general_finetuned_kirsch.pth models_kirsch/general_kirsch.pth