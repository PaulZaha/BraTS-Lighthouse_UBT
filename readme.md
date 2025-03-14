Step 1:
pip install -r requirements.txt

Step x:

get_data.py
This script loads the data into the synapse cache

training_cache_filedir = '188/141876188/BraTS-MEN-RT-Train-v2'
validation_cache_filedir = '485/143840485/BraTS-MEN-RT-Val-v1'

python training.py -traindir '188/141876188/BraTS-MEN-RT-Train-v2' -valdir '485/143840485/BraTS-MEN-RT-Val-v1'