import os
import numpy as np
import nibabel as nib
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def process_subject(subject, input_path, output_path):
    subject_path = os.path.join(input_path, subject)
    img_file = os.path.join(subject_path, subject + '_t1c.nii.gz')
    mask_file = os.path.join(subject_path, subject + '_gtv.nii.gz')

    # Skip if files are missing
    if not os.path.exists(img_file) or not os.path.exists(mask_file):
        print(f"Skipping {subject}: Missing files")
        return

    img_data = nib.load(img_file).get_fdata()
    mask_data = nib.load(mask_file).get_fdata()

    img_data = np.transpose(img_data, (2, 0, 1))
    mask_data = np.transpose(mask_data, (2, 0, 1))

    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, subject + '.npz')
    np.savez_compressed(output_file, imgs=img_data, gts=mask_data)

    print(f'Saved {output_file} with shape imgs: {img_data.shape}, gts: {mask_data.shape}')


def structure_brats_data(input_path, output_path, num_workers=12):
    subjects = os.listdir(input_path)
    os.makedirs(output_path, exist_ok=True)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(partial(process_subject, input_path=input_path, output_path=output_path), subjects)


def main():
    structure_brats_data(
        input_path='/mnt/Z/Repositories/BraTS/BraTS-Lighthouse_UBT/src_EfficientViT/data/raw_data_BraTS/BraTS-MEN-RT-Train-v2',
        output_path='/mnt/Z/Repositories/BraTS/BraTS-Lighthouse_UBT/src_EfficientViT/data/structured_data/BraTS',
        num_workers=12
    )


if __name__ == '__main__':
    main()
