import os
import numpy as np
from scipy.ndimage import convolve
from concurrent.futures import ProcessPoolExecutor

def build_kirsch_kernels():
    kernels = []
    for dz in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                kernel = np.full((3, 3, 3), -3, dtype=np.int32)
                kernel[1 + dz, 1 + dy, 1 + dx] = 5
                kernel[1, 1, 1] = 0
                kernels.append(kernel)
    return kernels

def apply_3d_kirsch(image, kernels):
    responses = []
    for kernel in kernels:
        response = convolve(image, kernel, mode='constant', cval=0)
        responses.append(response)
    response_stack = np.stack(responses, axis=0)
    edge_image = np.max(response_stack, axis=0)
    return edge_image

def process_file_cpu(input_filepath, output_filepath):
    data = np.load(input_filepath)
    image = data['imgs']
    
    kernels = build_kirsch_kernels()
    edge_image = apply_3d_kirsch(image, kernels)
    
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    np.savez(output_filepath, imgs=edge_image, gts=data['gts'])
    print(f"Verarbeitet: {input_filepath} -> {output_filepath}")

# Wrapper-Funktion, damit sie picklable ist
def process_file_cpu_wrapper(args):
    return process_file_cpu(*args)

def process_folder_parallel(input_folder, output_folder, num_workers=12):
    tasks = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.npz'):
                input_filepath = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                output_filepath = os.path.join(output_dir, file)
                tasks.append((input_filepath, output_filepath))
    print(tasks)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(process_file_cpu_wrapper, tasks)

if __name__ == "__main__":
    input_folder = "/mnt/Z/Repositories/BraTS/BraTS-Lighthouse_UBT/src_EfficientViT/data/structured_data"
    output_folder = "/mnt/Z/Repositories/BraTS/BraTS-Lighthouse_UBT/src_EfficientViT/data/structured_data_kirsch"
    
    process_folder_parallel(input_folder, output_folder)
