import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def inspect_nii(file):
    nii_img = nib.load(file)
    nii_data = nii_img.get_fdata()
    
    slices = np.linspace(0, nii_data.shape[2]-1, 10, dtype=int)
    
    fig, ax = plt.subplots(1, len(slices), figsize=(20, 4))
    fig.suptitle('10 Slices', fontsize=16)
    
    for i, slice_index in enumerate(slices):
        ax[i].imshow(nii_data[:, :, slice_index], cmap='gray', interpolation='none')
        ax[i].set_title(f'Slice {slice_index}')
        ax[i].axis('off')
    
    plt.savefig(f'{file}.png')
    plt.close()

def inspect_npz(file):

    data = np.load(file)
    
    imgs = data['imgs']
    
    slices = np.linspace(0, imgs.shape[0]-1, 10, dtype=int)
    
    fig, ax = plt.subplots(1, len(slices), figsize=(20, 4))
    fig.suptitle('10 Slices from npz', fontsize=16)
    
    for i, slice_index in enumerate(slices):
        ax[i].imshow(imgs[:, :, slice_index], cmap='gray', interpolation='none')
        ax[i].set_title(f'Slice {slice_index}')
        ax[i].axis('off')
    
    plt.savefig(f'{file}.png')
    plt.close()

def get_data_shape(file):
    nii_img = nib.load(file)
    nii_data = nii_img.get_fdata()
    return nii_data.shape

def main():
    # inspect_nii('/mnt/Z/Repositories/BraTS/BraTS-Lighthouse_UBT/src_EfficientViT/data/raw_data_BraTS/BraTS-MEN-RT-Train-v2/BraTS-MEN-RT-0004-1/BraTS-MEN-RT-0004-1_gtv.nii.gz')
    
    #inspect_nii('/mnt/Z/Repositories/BraTS/BraTS-Lighthouse_UBT/src_EfficientViT/data/raw_data_BraTS/BraTS-MEN-RT-Train-v2/BraTS-MEN-RT-0004-1/BraTS-MEN-RT-0004-1_t1c.nii.gz')
    
    inspect_npz('/mnt/Z/Repositories/BraTS/BraTS-Lighthouse_UBT/src_EfficientViT/data/structured_data/BraTS/BraTS-MEN-RT-0185-1.npz')

if __name__ == '__main__':
    main()
