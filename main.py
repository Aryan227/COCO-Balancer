import argparse
from utils import *

"""
Usage:
    python main.py --create_pruned_dataset --input_source_file input.json --output_file output.json
    python main.py --create_augmented_dataset --images_dir val2017 --input_source_file input_source.json --output_source_file output_source.json
    python main.py --create_augmented_dataset_doubled --images_dir val2017 --input_source_file input_source.json --output_source_file output_source.json
"""
def create_pruned_dataset(input_source_file, output_source_file,threshold):
    # Step 1: Create pruned dataset with rare classes
    create_pruned_dataset_rare_classes(input_file=input_source_file, output_file='output_rare.json', threshold=threshold)

    # Step 2: Create pruned dataset with common classes
    create_pruned_dataset_common_classes(original_file=input_source_file, input_file='output_rare.json', output_file='output_common.json', threshold=threshold)

    # Step 3: Combine datasets
    combine_datasets(original_file=input_source_file, input_file_1='output_rare.json', input_file_2='output_common.json', output_file=output_source_file)


#This will keep your dataset size constant
def create_augmented_dataset(images_dir, input_source_file, output_source_file):
    # Step 1: Create augmented dataset
    augment_dataset(source_images = images_dir, input_json = input_source_file, output_json = output_source_file)


#This will double your dataset size
def create_augmented_dataset_doubled(images_dir, input_source_file, output_source_file):
    # Step 1: Create augmented dataset
    augment_dataset(source_images = images_dir, input_json = input_source_file, output_json = 'output_augmented.json')

    #Step 2: Combine Datasets
    combine_datasets(original_file=input_source_file, input_file_1=input_source_file, input_file_2='output_augmented.json', output_file=output_source_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--create_pruned_dataset', action='store_true', help='Create pruned dataset')
    parser.add_argument('--create_augmented_dataset', action='store_true', help='Create augmented dataset of same size')
    parser.add_argument('--create_augmented_dataset_doubled', action='store_true', help='Create augmented dataset of double size')

    parser.add_argument('--input_source_file', type=str, help='Path to input source file')
    parser.add_argument('--output_source_file', type=str, help='Path to output source file')
    parser.add_argument('--images_dir', type=str, help='Path to source images')

    parser.add_argument('--threshold', type=int, help='Threshold to determine how many images we need from a certain class')

    args = parser.parse_args()

    if args.create_pruned_dataset and args.input_source_file and args.output_source_file and args.threshold:
        create_pruned_dataset(args.input_source_file, args.output_source_file, args.threshold)
    
    if args.create_augmented_dataset and args.images_dir and args.input_source_file and args.output_source_file:
        create_augmented_dataset(args.images_dir, args.input_source_file, args.output_source_file)
    
    if args.create_augmented_dataset_doubled and args.images_dir and args.input_source_file and args.output_source_file:
        create_augmented_dataset_doubled(args.images_dir, args.input_source_file, args.output_source_file)
