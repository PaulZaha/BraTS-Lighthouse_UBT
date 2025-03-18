Step 1:
pip install -r requirements.txt

Step x:
go to mednext folder
pip install -e .

get_data.py
This script loads the data into the synapse cache

get_dataset_structure.py
copies downloaded data into the folder structure

Step x:
copy data into data/raw folder

Step x:
create_dataset_json.py

Step x:
mednextv1_plan_and_preprocess -t 500 -pl3d ExperimentPlanner3D_v21_customTargetSpacing_1x1x1
mednextv1_plan_and_preprocess -t 501 -pl3d ExperimentPlanner3D_v21_customTargetSpacing_1x1x1 (for testing with only 10 images)
mednextv1_plan_and_preprocess -t 501 -pl2d None (for testing with only 10 images)

this worked?
mednextv1_plan_and_preprocess -t 501 -pl3d ExperimentPlanner3D_v21_customTargetSpacing_1x1x1 -pl2d ExperimentPlanner2D_v21_customTargetSpacing_1x1x1

mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_S_kernel3 Task501 0 -p nnUNetPlansv2.1_trgSp_1x1x1
mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_S_kernel3 Task501 0 (for testing with only 10 images)

this worked?
mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_L_kernel5 Task501 0 -p nnUNetPlansv2.1_trgSp_1x1x1