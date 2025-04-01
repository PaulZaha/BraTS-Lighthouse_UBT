import os
import pandas as pd
import argparse
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("trainpath")
parser.add_argument("valpath")
parser.add_argument("trainpathkirsch")
parser.add_argument("valpathkirsch")
args = parser.parse_args()
trainfiles=pd.read_csv(args.trainpath)
valfiles=pd.read_csv(args.valpath)
trainfiles_kirsch=pd.read_csv(args.trainpathkirsch)
valfiles_kirsch=pd.read_csv(args.valpathkirsch)

modalities={
    "CT":r"^CT",
    "BraTS":r"^BraTS"
}

os.makedirs("data/dataset_csvs/modalities3D", exist_ok=True)
os.makedirs("data/dataset_csvs/modalities3D_kirsch", exist_ok=True)

for modality, prefix_regex in modalities.items():
    pd.DataFrame(trainfiles[trainfiles['file'].str.extract(r'([^/]+)$')[0].str.match(prefix_regex)], columns=['file']).to_csv(f'data/dataset_csvs/modalities3D/{modality}.train.csv', index=False)
    pd.DataFrame(valfiles[valfiles['file'].str.extract(r'([^/]+)$')[0].str.match(prefix_regex)], columns=['file']).to_csv(f'data/dataset_csvs/modalities3D/{modality}.val.csv', index=False)

    pd.DataFrame(trainfiles_kirsch[trainfiles_kirsch['file'].str.extract(r'([^/]+)$')[0].str.match(prefix_regex)], columns=['file']).to_csv(f'data/dataset_csvs/modalities3D_kirsch/{modality}_kirsch.train.csv', index=False)
    pd.DataFrame(valfiles_kirsch[valfiles_kirsch['file'].str.extract(r'([^/]+)$')[0].str.match(prefix_regex)], columns=['file']).to_csv(f'data/dataset_csvs/modalities3D_kirsch/{modality}_kirsch.val.csv', index=False)

