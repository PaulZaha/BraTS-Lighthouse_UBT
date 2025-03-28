import os
import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import convolve as gpu_convolve
from scipy.ndimage import convolve as cpu_convolve
from concurrent.futures import ThreadPoolExecutor, as_completed

def build_kirsch_kernels():
    kernels = []
    for dz in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                kernel = cp.full((3, 3, 3), -3, dtype=cp.int32)
                kernel[1 + dz, 1 + dy, 1 + dx] = 5
                kernel[1, 1, 1] = 0 
                kernels.append(kernel)
    return kernels

def apply_3d_kirsch_gpu(image, kernels, stream):
    with stream:
        responses = []
        for kernel in kernels:
            response = gpu_convolve(image, kernel, mode='constant', cval=0)
            responses.append(response)
        response_stack = cp.stack(responses, axis=0)
        edge_image = cp.max(response_stack, axis=0)
    return edge_image

def process_file(input_filepath, output_filepath, kernels):
    if os.path.exists(output_filepath):
        print(f"Datei {output_filepath} existiert bereits, überspringe Verarbeitung.")
        return

    try:
        stream = cp.cuda.Stream(non_blocking=True)
        data = np.load(input_filepath)
        image = data['imgs']
        cp_image = cp.asarray(image)
        
        edge_image = apply_3d_kirsch_gpu(cp_image, kernels, stream)
        stream.synchronize()
        edge_image_np = cp.asnumpy(edge_image)
        
        print(f"GPU-Berechnung für {input_filepath} erfolgreich.")
    except cp.cuda.memory.OutOfMemoryError:
        pass
        # print(f"CUDA Out-of-Memory bei {input_filepath}. Wechsle zu CPU-Berechnung.")
        # # CPU-Berechnung: Konvertiere die Kernel in numpy-Arrays
        # data = np.load(input_filepath)
        # image = data['imgs']
        # responses = []
        # for kernel in kernels:
        #     kernel_cpu = cp.asnumpy(kernel)
        #     response = cpu_convolve(image, kernel_cpu, mode='constant', cval=0)
        #     responses.append(response)
        # response_stack = np.stack(responses, axis=0)
        # edge_image_np = np.max(response_stack, axis=0)
    
    try:
        # Speichern der Ergebnisse
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        np.savez_compressed(output_filepath, imgs=edge_image_np, gts=data['gts'])
        print(f"Konvolvierte Datei {output_filepath} gespeichert.")
    except:
        pass

    try:
        del cp_image, edge_image
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass

def process_folder_parallel(input_folder, output_folder, num_workers=12):
    kernels = build_kirsch_kernels()
    file_list = []
    
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.npz'):
                input_filepath = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                output_filepath = os.path.join(output_dir, file)
                file_list.append((input_filepath, output_filepath))
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_file, inp, out, kernels)
            for inp, out in file_list
        ]
        for future in as_completed(futures):
            future.result()

if __name__ == "__main__":
    input_folder = "/mnt/Z/Repositories/BraTS/BraTS-Lighthouse_UBT/src_EfficientViT/data/structured_data"
    output_folder = "/mnt/Z/Repositories/BraTS/BraTS-Lighthouse_UBT/src_EfficientViT/data/structured_data_kirsch"
    
    process_folder_parallel(input_folder, output_folder, num_workers=1)
