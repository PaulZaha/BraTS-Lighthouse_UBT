Step 1:
pip install -r requirements.txt

Step x:

get_data.py
This script loads the data into the synapse cache

Step x:
copy data into data/raw folder

Step x:
preprocess the data into the needed form for MedNeXt model

Step x:
python training.py --images_dir /mnt/Z/Repositories/BraTS-Lighthouse_UBT/data/nnUNet_raw_data_base/imagesTr --masks_dir /mnt/Z/Repositories/BraTS-Lighthouse_UBT/data/nnUNet_raw_data_base/labelsTr
