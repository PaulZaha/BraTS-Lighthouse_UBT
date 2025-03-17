import os
import re
import shutil

def preprocess_data():
    data_dir = os.path.join(os.getcwd(),'..','data','raw','BraTS-MEN-RT-Train-v2')
    gtv_target = os.path.join(os.getcwd(),'..','data','nnUNet_raw_data_base','imagesTr')
    t1c_target = os.path.join(os.getcwd(),'..','data','nnUNet_raw_data_base','labelsTr')
    
    os.makedirs(gtv_target, exist_ok=True)
    os.makedirs(t1c_target, exist_ok=True)

    for subfolder in os.listdir(data_dir):
        subfolder_path = os.path.join(data_dir, subfolder)
        
        # Überprüfe, ob es sich um einen Ordner handelt
        if os.path.isdir(subfolder_path):
            match = re.search(r'BraTS-MEN-RT-(\d+)-\d+', subfolder)
            
            if match:
                id_number = match.group(1)  # Extrahiere die ID (z.B. 0004, 0009)
                new_filename = f"BRATS_{id_number}_0000.nii.gz"
                
                # Suche die _gtv und t1c Dateien
                for file in os.listdir(subfolder_path):
                    file_path = os.path.join(subfolder_path, file)
                    
                    if file.endswith("_gtv.nii.gz"):
                        shutil.copy(file_path, os.path.join(gtv_target, new_filename))
                        print(f"Kopiert: {file_path} → {gtv_target}/{new_filename}")

                    elif "t1c" in file and file.endswith(".nii.gz"):
                        shutil.copy(file_path, os.path.join(t1c_target, new_filename))
                        print(f"Kopiert: {file_path} → {t1c_target}/{new_filename}")


def main():
    preprocess_data()

if __name__ == '__main__':
    main()