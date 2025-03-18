from ..models.mednext.nnunet_mednext.dataset_conversion.utils import generate_dataset_json

def main():
    generate_dataset_json(
        output_file='/mnt/Z/Repositories/BraTS/BraTS-Lighthouse_UBT/data/nnUNet_raw_data_base/nnUNet_raw_data/Task500/dataset.json',
        imagesTr_dir='/mnt/Z/Repositories/BraTS/BraTS-Lighthouse_UBT/data/nnUNet_raw_data_base/nnUNet_raw_data/Task500/imagesTr',
        imagesTs_dir=None,
        modalities=('T1C'),
        labels={0: 'Background',1: 'GTV'},
        dataset_name='BraTS_UBT'
    )

if __name__ == '__main__':
    main()