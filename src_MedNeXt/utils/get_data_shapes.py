import os
import numpy as np

def print_npz_shapes(folder_path):
    # Durchlaufe alle Dateien im angegebenen Ordner
    for filename in os.listdir(folder_path):
        if filename.endswith(".npz"):
            file_path = os.path.join(folder_path, filename)
            print(f"Datei: {filename}")
            # Lade die .npz-Datei
            data = np.load(file_path)
            # Iteriere über alle Arrays in der .npz-Datei
            for key in data.files:
                shape = data[key].shape
                print(f"  Array '{key}' hat Shape: {shape}")
            print("-" * 40)

# Beispiel-Aufruf: passe den Pfad zum gewünschten Ordner an
if __name__ == "__main__":
    folder_path = "/mnt/Z/Repositories/BraTS/BraTS-Lighthouse_UBT/data/nnUNet_raw_data_base/nnUNet_cropped_data/Task501"  # Ersetze dies durch den tatsächlichen Ordnerpfad
    print_npz_shapes(folder_path)
