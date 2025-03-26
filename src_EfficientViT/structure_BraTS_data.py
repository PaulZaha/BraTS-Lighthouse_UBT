import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt



def structure_brats_data(input_path, output_path):
    subjects = os.listdir(input_path)
    
    # Sicherstellen, dass output_path existiert
    os.makedirs(output_path, exist_ok=True)
    
    for subject in subjects:
        subject_path = os.path.join(input_path, subject)

        img_file = os.path.join(subject_path, subject + '_t1c.nii.gz')
        mask_file = os.path.join(subject_path, subject + '_gtv.nii.gz')
        

        img_data = nib.load(img_file).get_fdata()
        mask_data = nib.load(mask_file).get_fdata()
        

        img_data = np.transpose(img_data, (2, 0, 1))
        mask_data = np.transpose(mask_data, (2, 0, 1))
        

        output_file = os.path.join(output_path, subject + '.npz')
        np.savez(output_file, imgs=img_data, gts=mask_data)
        print(f'Saved {output_file} with shape imgs: {img_data.shape}, gts: {mask_data.shape}')

def main():
    
    structure_brats_data(
        input_path='/mnt/Z/Repositories/BraTS/BraTS-Lighthouse_UBT/src_EfficientViT/data/raw_data_BraTS/BraTS-MEN-RT-Train-v2',
        output_path='/mnt/Z/Repositories/BraTS/BraTS-Lighthouse_UBT/src_EfficientViT/data/structured_data/BraTS'
    )

if __name__ == '__main__':
    main()
