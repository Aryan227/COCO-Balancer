import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import json
import cv2
import os
import matplotlib.patches as patches
import argparse

from pycocotools.coco import COCO

#This file contains helper functions to validate your processes and also to visualize some results.
#All the helper functions can be run in console as a command

"""
Usage:
    python helper.py --visualize_class_occurrences --input_source_file val_pruned_augmented.json
    python helper.py --count_files_in_directory --images_dir val2017
    python helper.py --visualize_class_occurrences --input_source_file val_pruned_augmented.json
    python helper.py --count_files_in_directory --images_dir val2017
"""

#Use this to validate if the files have been downloaded properly
def count_files_in_directory(input_dir):
    """
    Count the number of files in a directory.

    Args:
        input_dir (str): Path to the directory. This redirects to source images: val2017

    Example:
        >>> count_files_in_directory('/content/val2017')
    """
    count = 0

    # Iterate directory
    for path in os.listdir(input_dir):
        # check if current path is a file
        if os.path.isfile(os.path.join(input_dir, path)):
            count += 1

    print(count)


#This can be used to visualize your class ocuurences so one can do an eye test on the how well the dataset is balanced
def visualize_class_occurrences(input_file):
    """
    Visualize class occurrences as a pie chart.

    Args:
        input_file (str): Path to the input JSON file containing COCO annotations.

    This function reads COCO annotations from the specified JSON file, calculates
    the occurrences of each class, and visualizes it as a pie chart.

    Example:
        >>> visualize_class_occurrences('pruned_dataset_val_without_door.json')
    """
    ann_file = input_file
    coco = COCO(ann_file)
    class_occurrences = {}

    for ann in coco.dataset['annotations']:
        category_id = ann['category_id']
        category_name = coco.loadCats(category_id)[0]['name']

        if category_name not in class_occurrences:
            class_occurrences[category_name] = 1
        else:
            class_occurrences[category_name] += 1

    df_class_occurrences = pd.DataFrame(list(class_occurrences.items()), columns=['Class', 'Occurrences'])
    df_class_occurrences = df_class_occurrences.sort_values(by='Occurrences', ascending=False)
    df_class_occurrences = df_class_occurrences.reset_index(drop=True)

    total_images = len(coco.dataset['images'])
    print('Total images now are',total_images)

    df_class_occurrences['Percent of Images'] = df_class_occurrences['Occurrences'] / total_images

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.pie(df_class_occurrences['Percent of Images'], labels=df_class_occurrences['Class'], autopct='%1.1f%%', startangle=90)
    ax.axis('equal')

    # Save the pie chart as an image
    output_image_path = input_file.replace('.json', '_class_occurrences.png')
    plt.savefig(output_image_path)
    #print(f'Pie chart saved as {output_image_path}')

    # Display the pie chart (optional)
    # plt.show()



#This helps you visualize the images with annotations
def visualize_images_with_bbox(source_images, input_json, threshold):
    """
    Visualize images with bounding boxes.

    Args:
        source_images (str): Path to the source images folder.
        input_json (str): Path to the input JSON file containing COCO annotations.
        threshold (int): Number of images to visualize.

    This function loads COCO annotations from the specified JSON file, retrieves the
    corresponding images, and visualizes them with bounding boxes. It saves the images
    in a folder named 'visualized_images' created in the current working directory.

    Example:
        >>> visualize_images_with_bbox('val2017', 'pruned_dataset_val_without_door.json', 10)
    """
    # Load pruned_dataset.json
    with open(input_json, 'r') as f:
        pruned_dataset = json.load(f)

    images = pruned_dataset['images']
    annotations = pruned_dataset['annotations']

    # Define a folder to store visualized images
    output_folder = 'visualized_images'
    os.makedirs(output_folder, exist_ok=True)

    # Define a function to visualize images with bounding boxes
    def visualize_image_with_bbox(image_path, annotations):
        # Load the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create a figure and axis
        fig, ax = plt.subplots(1)

        # Display the image
        ax.imshow(image)

        # Iterate through annotations and draw bounding boxes
        for ann in annotations:
            bbox = ann['bbox']
            category_id = ann['category_id']

            # Create a Rectangle patch
            x, y, w, h = bbox
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)

        # Set axis limits
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(0, image.shape[0])

        # Save the plot as an image
        output_image_path = os.path.join(output_folder, os.path.basename(image_path))
        plt.savefig(output_image_path)
        plt.close(fig)

    # Iterate through the specified number of images and visualize each with its annotations
    for img_info in images[:threshold]:
        image_path = os.path.join(source_images, img_info['file_name'])
        image_annotations = [ann for ann in annotations if ann['image_id'] == img_info['id']]
        visualize_image_with_bbox(image_path, image_annotations)


#This function measures the balance of the dataset
def shannon_entropy(input_json):
    """
    Calculate the shannon entropy of your dataset. This is used as a metric to calculate "balance"
    Credit: https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/shannon.htm
    Range: 0 to 1 (inclusive)
           1 represents that your classes are equally distributed, 0 represnts presence of only 1 class.

    Args:
        input_json (str): Path to the input JSON file containing COCO annotations.

    Example:
        >>> shannon_entropy('pruned_dataset.json')
    """
    ann_file = input_json
    coco = COCO(ann_file)
    class_occurrences = {}

    for ann in coco.dataset['annotations']:
        category_id = ann['category_id']
        category_name = coco.loadCats(category_id)[0]['name']

        if category_name not in class_occurrences:
            class_occurrences[category_name] = 1
        else:
            class_occurrences[category_name] += 1

    df_class_occurrences = pd.DataFrame(list(class_occurrences.items()), columns=['Class', 'Occurrences'])
    df_class_occurrences = df_class_occurrences.sort_values(by='Occurrences', ascending=False)
    df_class_occurrences = df_class_occurrences.reset_index(drop=True)

    n = df_class_occurrences['Occurrences'].sum()
    k = len(df_class_occurrences)

    H = -sum((count/n) * np.log(count/n) for count in df_class_occurrences['Occurrences'])  # Shannon entropy
    entropy_measure =  H / np.log(k)
    print(entropy_measure)


#This function helps validate if augmentations/pruning worked properly
def compare_json(file1, file2):
    """
    This function compares 2 json files.

    Args:
        file1 (str): Path to the input JSON file containing COCO annotations.
        file2 (str): Path to the input JSON file containing COCO annotations.
    
    Example:
        >>> compare_json('pruned_dataset_val.json', 'pruned_dataset_val_augmented.json')
    """
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        json1 = json.load(f1)
        json2 = json.load(f2)

    if json1==json2:
        print("The JSON files are identical.")
    else:
        print("The JSON files are not identical.")
    
    return json1 == json2



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--count_files_in_directory', action='store_true', help='Counts files in a folder')
    parser.add_argument('--visualize_class_occurrences', action='store_true', help='Create a pie-chart for class counts and outputs total images')
    parser.add_argument('--visualize_images_with_bbox', action='store_true', help='Create a folder of images with bounding boxes')
    parser.add_argument('--shannon_entropy', action='store_true', help='Outputs the shannon entropy measure of a dataset')
    parser.add_argument('--compare_json', action='store_true',help='Outputs if two jsons are similar or not')

    parser.add_argument('--input_source_file', type=str, help='Path to input source file (json)')
    parser.add_argument('--images_dir', type=str, help='Path to source images')
    parser.add_argument('--threshold', type=int, help='Threshold to determine uptil which image you want to visualize the bounding boxes i.e.[:threshold]')
    parser.add_argument('--input_source_file_2', type=str, help='Path to second input source file (json)')

    args = parser.parse_args()

    if args.count_files_in_directory and args.images_dir:
        count_files_in_directory(args.images_dir)

    if args.visualize_class_occurrences and args.input_source_file:
        visualize_class_occurrences(args.input_source_file)

    if args.visualize_images_with_bbox and args.images_dir and args.input_source_file and args.threshold:
        visualize_images_with_bbox(args.images_dir,args.input_source_file,args.threshold)

    if args.shannon_entropy and args.input_source_file:
        shannon_entropy(args.input_source_file)

    if args.compare_json and args.input_source_file and args.input_source_file_2:
        compare_json(args.input_source_file,args.input_source_file_2)
