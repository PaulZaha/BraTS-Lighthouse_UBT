import glob
import random
import os
import pandas as pd
from pathlib import Path

BASE_PATH = Path(__file__).parent

modalities = glob.glob((BASE_PATH / "data/structured_data/*").absolute().as_posix())
print(modalities)
train_files = []
val_files = []

for m in modalities:
    if os.path.isfile(m):
        continue
    print(m)
    # .npz-Dateien direkt in m hinzufügen
    npzs_direct = glob.glob(m + "/*.npz")
    random.shuffle(npzs_direct)
    l = len(npzs_direct)
    split_index = int(l * 0.9)
    train_files.extend(npzs_direct[:split_index])
    val_files.extend(npzs_direct[split_index:])
    
    # Nun die Unterordner abarbeiten
    for subdir in glob.glob(m + "/*"):
        if os.path.isfile(subdir):
            continue
        print(subdir)
        npzs = glob.glob(subdir + "/*.npz")
        random.shuffle(npzs)
        l = len(npzs)
        split_index = int(l * 0.9)
        train_files.extend(npzs[:split_index])
        val_files.extend(npzs[split_index:])

(pd.DataFrame(train_files, columns=['file'])
    .to_csv(BASE_PATH / 'data/dataset_csvs/train.csv', index=False))
(pd.DataFrame(val_files, columns=['file'])
    .to_csv(BASE_PATH / 'data/dataset_csvs/val.csv', index=False))
(pd.DataFrame(train_files + val_files, columns=['file'])
    .to_csv(BASE_PATH / 'data/dataset_csvs/fulldataset.csv', index=False))


modalities = glob.glob((BASE_PATH / "data/structured_data_kirsch/*").absolute().as_posix())
print(modalities)
train_files = []
val_files = []

for m in modalities:
    if os.path.isfile(m):
        continue
    print(m)
    # .npz-Dateien direkt in m hinzufügen
    npzs_direct = glob.glob(m + "/*.npz")
    random.shuffle(npzs_direct)
    l = len(npzs_direct)
    split_index = int(l * 0.9)
    train_files.extend(npzs_direct[:split_index])
    val_files.extend(npzs_direct[split_index:])
    
    # Nun die Unterordner abarbeiten
    for subdir in glob.glob(m + "/*"):
        if os.path.isfile(subdir):
            continue
        print(subdir)
        npzs = glob.glob(subdir + "/*.npz")
        random.shuffle(npzs)
        l = len(npzs)
        split_index = int(l * 0.9)
        train_files.extend(npzs[:split_index])
        val_files.extend(npzs[split_index:])

(pd.DataFrame(train_files, columns=['file'])
    .to_csv(BASE_PATH / 'data/dataset_csvs/train_kirsch.csv', index=False))
(pd.DataFrame(val_files, columns=['file'])
    .to_csv(BASE_PATH / 'data/dataset_csvs/val_kirsch.csv', index=False))
(pd.DataFrame(train_files + val_files, columns=['file'])
    .to_csv(BASE_PATH / 'data/dataset_csvs/fulldataset_kirsch.csv', index=False))