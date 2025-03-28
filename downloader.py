import os
import zipfile
import gdown

#=Dataset-download======================================================================================================
dataset_id = "1M02Tyj-I3LLwPKse3QlUrjTHf7DHkjIf"
dataset_zip = "asl_dataset.zip"
dataset_dir = "datasets/asl_main/"

if not os.path.exists(dataset_dir):
    print("Dataset not found locally. Downloading from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={dataset_id}", dataset_zip, quiet=False)

    print("Extracting dataset...")
    with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
        zip_ref.extractall("datasets/")

    os.remove(dataset_zip)
    print("Dataset download and extraction complete.")
else:
    print("Dataset already exists. Skipping download.")


#=Model-downloads=======================================================================================================
file_ids = {
    "resnet50.h5": "1-dRJSRC401mvnm_TtCTBaTpq2ZZiCATd",
    "og_cnn.h5": "1IDKmRIDm9ocCMDNMoQDgDskkBoTleKMZ",
}

model_dir = "models/asl/"
os.makedirs(model_dir, exist_ok=True)

for filename, file_id in file_ids.items():
    output_path = os.path.join(model_dir, filename)

    if not os.path.exists(output_path):
        print(f"{filename} not found locally. Downloading from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
        print(f"{filename} download complete.")
    else:
        print(f"{filename} already exists. Skipping download.")

print("All downloads complete.")
