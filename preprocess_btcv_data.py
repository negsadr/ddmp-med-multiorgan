import argparse
import glob
import numpy as np
import nibabel as nib
import os
from tqdm import tqdm
import torchio as tio


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess CT datasets.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where the outputs will be saved")
    return parser.parse_args()


def create_dirs(output_dir):
    dirs = {
        "img": os.path.join(output_dir, "img"),
        "label": os.path.join(output_dir, "label"),
    }
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    return dirs


def load_data_list(data_dir, modality):
    return sorted(glob.glob(os.path.join(data_dir,  f"{modality}", "*.nii.gz")))


def preprocess_and_save(subject, output_dirs, img_names):
    print (subject)
    for modality, img in subject.items():
        print(modality, img)
        modality_key = modality.replace("img_", "")
        if modality_key != "label":  # Skip mask for intensity rescaling
            transform = tio.RescaleIntensity((-1, 1))
            img = transform(img)
        img.save(os.path.join(output_dirs[modality_key], img_names[modality_key]))


def preprocess_label(image, label_path, affine):
    img = nib.load(image).get_fdata()
    label = nib.load(label_path).get_fdata().astype(np.uint8)
    label = np.where(label == 0, img, label)
    nib.save(nib.Nifti1Image(label, affine), label_path)


def main():
    args = parse_args()
    output_dirs = create_dirs(args.output_dir)

    modalities = ["img", "label"]
    data_lists = {modality: load_data_list(args.data_dir, modality) for modality in modalities}

    # Preprocess and crop
    for idx in tqdm(range(len(data_lists["img"]))):
        img_names = {modality: os.path.basename(data_lists[modality][idx]) for modality in modalities}
        subject = tio.Subject(
            img_img=tio.ScalarImage(data_lists["img"][idx]),
            img_label=tio.LabelMap(data_lists["label"][idx])
        )
        transform = tio.CropOrPad((128, 128, 128))
        subject = transform(subject)
        preprocess_and_save(subject, output_dirs, img_names)

    # Preprocess mask separately
    for image_path, label_path in tqdm(zip(data_lists["img"], data_lists["label"])):
        preprocess_label(image_path, label_path, nib.load(label_path).affine)

    print("COMPLETE!")


if __name__ == "__main__":
    main()