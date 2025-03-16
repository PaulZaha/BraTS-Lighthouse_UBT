import os
import torch
import argparse
from models.MedNeXt.nnunet_mednext import create_mednext_v1
from models.MedNeXt.nnunet_mednext.training.network_training.MedNeXt.nnUNetTrainerV2_MedNeXt import nnUNetTrainerV2_MedNeXt_L_kernel5

parser = argparse.ArgumentParser()
parser.add_argument(
    "-traindir", type=str, default="",
    help="training directory from syncache"
)
parser.add_argument(
    "-valdir", type=str, default="",
    help="validation directory from syncache"
)
args = parser.parse_args()
traindir = args.traindir
valdir = args.valdir


def init_model(input_channels: int, num_classes: int, model_id: str, kernel_size: int, deep_supervision: bool):
    model = create_mednext_v1(
        num_input_channels=input_channels,
        num_classes=num_classes,
        model_id=model_id,
        kernel_size=kernel_size,
        deep_supervision=deep_supervision
    )
    return model

def init_trainer(plans_file, fold):
    trainer = nnUNetTrainerV2_MedNeXt_L_kernel5(
        plans_file=plans_file,
        fold=fold
    )
    return trainer


def main():
    input_channels = 100
    num_classes = 2
    model_id = 'L'
    kernel_size = 5
    deep_supervision = True
    
    # model = init_model(
    #     input_channels=input_channels,
    #     num_classes=num_classes,
    #     model_id=model_id,
    #     kernel_size=kernel_size,
    #     deep_supervision=deep_supervision
    # )
    # print(model)

    plans_file = xx
    fold = xx
    trainer = init_trainer(
        plans_file=plans_file,
        fold=fold
    )
    print(trainer)

if __name__ == '__main__':
    main()