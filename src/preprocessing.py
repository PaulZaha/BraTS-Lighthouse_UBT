from src.models.MedNeXt.nnunet_mednext.dataset_conversion.utils import generate_dataset_json


def main():
    generate_dataset_json(
        output_file="/mnt/Z/Repositories/BraTS-Lighthouse_UBT/data/preprocessed",
        imagesTr_dir="~/.synapseCache/188/141876188/BraTS-MEN-RT-Train-v2",

    )

if __name__ == '__main__':
    main()