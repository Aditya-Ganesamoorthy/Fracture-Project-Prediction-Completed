import os
import cv2
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Data Augmentation')
    parser.add_argument('--input_img', type=str, required=True, help='Path to input images')
    parser.add_argument('--output_img', type=str, required=True, help='Path to output images')
    parser.add_argument('--input_label', type=str, required=True, help='Path to input labels')
    parser.add_argument('--output_label', type=str, required=True, help='Path to output labels')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    input_imgdir = args.input_img
    output_imgdir = args.output_img
    input_labeldir = args.input_label
    output_labeldir = args.output_label

    os.makedirs(output_imgdir, exist_ok=True)
    os.makedirs(output_labeldir, exist_ok=True)

    print("🚀 Starting image augmentation...")

    count = 0

    # Process images
    for name in os.listdir(input_imgdir):

        if not name.endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(input_imgdir, name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ Skipping corrupted image: {name}")
            continue

        # Augmentation
        contrast = 1.2
        brightness = 30
        img_aug = cv2.addWeighted(img, contrast, img, 0, brightness)

        # Save original
        cv2.imwrite(os.path.join(output_imgdir, name), img)

        # Save augmented
        cv2.imwrite(os.path.join(output_imgdir, "aug_" + name), img_aug)

        count += 1
        if count % 1000 == 0:
            print(f"Processed {count} images...")

    print(f"✅ Image augmentation completed ({count} images processed)")

    print("🚀 Processing labels...")

    label_count = 0

    # Process labels
    for name in os.listdir(input_labeldir):

        if not name.endswith(".txt"):
            continue

        label_path = os.path.join(input_labeldir, name)

        with open(label_path, "r") as f:
            lines = f.readlines()

        # Save original
        with open(os.path.join(output_labeldir, name), "w") as f:
            f.writelines(lines)

        # Save augmented
        with open(os.path.join(output_labeldir, "aug_" + name), "w") as f:
            f.writelines(lines)

        label_count += 1

    print(f"✅ Label augmentation completed ({label_count} labels processed)")
    print("🎉 All augmentation done successfully!")