import os
import json
import argparse

def convert_coco_to_yolo(input_file, output_dir): #output_dir should be called ../labels
    """
    Convert COCO annotations to YOLO format.

    Args:
        input_file (str): Path to the input JSON file containing COCO annotations.
        output_dir (str): Path to the output directory where YOLO labels will be saved.

    This function converts COCO annotations to YOLO format. It reads the annotations
    from the specified JSON file, calculates YOLO format values, and saves the labels
    in the specified output directory.
    """

    with open(input_file, 'r') as f:
        pruned_dataset_1 = json.load(f)

    coco_annotations = pruned_dataset_1['annotations']

    for annotation in coco_annotations:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        bbox = annotation['bbox']
        img_info = coco.loadImgs(image_id)[0]

        img_width = img_info['width']
        img_height = img_info['height']

        x_center = (bbox[0] + bbox[2] / 2) / img_width
        y_center = (bbox[1] + bbox[3] / 2) / img_height
        width = bbox[2] / img_width
        height = bbox[3] / img_height

        label_line = f"{category_id - 1} {x_center} {y_center} {width} {height}\n"

        image_filename = img_info['file_name'].split('.')[0]  # Remove file extension
        label_filename = f"{output_folder}/{image_filename}.txt"

        with open(label_filename, 'a') as label_file:
            label_file.write(label_line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--convert_coco_to_yolo', action='store_true', help='Create dataset with additonal file')

    parser.add_argument('--input_source_file', type=str, help='Path to input source file')
    parser.add_argument('--output_dir', type=str, help='Path to output dir (Should be of the type ../labels)')

    args = parser.parse_args()

    if args.convert_coco_to_yolo and args.input_source_file and args.output_dir:
        convert_coco_to_yolo(args.input_source_file, args.output_dir)

