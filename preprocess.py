import os
from PIL import Image
from tqdm import tqdm

INPUT_DIR = "final_dataset_v3"
OUTPUT_DIR = "data_processed"
IMAGE_SIZE = (256, 256)


def process_image(input_path, output_path):
    try:
        img = Image.open(input_path).convert("RGB")
        img = img.resize(IMAGE_SIZE)
        img.save(output_path)
        return True
    except Exception:
        # corrupted or unreadable image — skip
        return False


def process_split(split):
    input_split_path = os.path.join(INPUT_DIR, split)
    output_split_path = os.path.join(OUTPUT_DIR, split)
    os.makedirs(output_split_path, exist_ok=True)

    for label in ["real", "fake"]:
        input_label_path = os.path.join(input_split_path, label)
        output_label_path = os.path.join(output_split_path, label)
        os.makedirs(output_label_path, exist_ok=True)

        if not os.path.exists(input_label_path):
            continue

        files = os.listdir(input_label_path)

        print(f"Processing {split}/{label} …")

        for f in tqdm(files):
            in_file = os.path.join(input_label_path, f)
            out_file = os.path.join(output_label_path, f)

            process_image(in_file, out_file)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split in ["train", "val", "test"]:
        process_split(split)

    print("\nPreprocessing complete. Cleaned + resized dataset stored in 'data_processed/' ")


if __name__ == "__main__":
    main()
