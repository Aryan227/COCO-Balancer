import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import json
import cv2
import os
import matplotlib.patches as patches

from pycocotools.coco import COCO
from albumentations import BboxParams, Compose, Crop, OneOf, VerticalFlip, HorizontalFlip, Resize, Rotate

#This file contains all the necessary functions needed to create a balance COCO dataset and also augment it

def create_pruned_dataset_rare_classes(input_file, output_file, threshold):
    """
    Process COCO annotations to create a pruned dataset.
    Logic: Get common and rare classes based on their count in the COCO annotations
           Include all images that contain any rare classes into the pruned dataset

    Args:
        input_file (str): Path to the input COCO annotations file.
        output_file (str): Path to save the pruned dataset in COCO format.
        threshold (int): Threshold for categorizing common and rare classes.

    Returns:
        None
    """
    coco = COCO(input_file)

    pruned_dataset = []

    #Get common and rare classes
    common_classes = []
    rare_classes = []

    for cat_id, category in coco.cats.items():
        if len(coco.getImgIds(catIds=[cat_id])) > threshold:
            common_classes.append(cat_id)
        else:
            rare_classes.append(cat_id)


    annotations = []

    # Ensure rare image ids go first and then common image ids
    image_ids_with_rare_classes = []

    for img_id in coco.getImgIds():
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        contains_rare_class = any(ann['category_id'] in rare_classes for ann in anns)

        if contains_rare_class:
            image_ids_with_rare_classes.append(img_id)

    image_ids_without_rare_classes = [img_id for img_id in coco.getImgIds() if img_id not in image_ids_with_rare_classes]

    image_ids = image_ids_with_rare_classes + image_ids_without_rare_classes

    for img_id in image_ids:
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = coco.loadAnns(ann_ids)

        contains_rare_class = any(ann['category_id'] in rare_classes for ann in anns)

        if contains_rare_class:
            pruned_dataset.append(img_info)

            for ann in anns:
                category_id = ann['category_id']
                bbox = ann['bbox']
                area = ann['area']
                iscrowd = ann['iscrowd']
                annotation_id = ann['id']

                annotations.append({
                    'image_id': img_id,
                    'category_id': category_id,
                    'bbox': bbox,
                    'area': area,
                    'iscrowd': iscrowd,
                    'id': annotation_id
                })

    coco_format_annotations = {
        "images": pruned_dataset,
        "annotations": annotations,
        "categories": coco.dataset["categories"]
    }

    with open(output_file, 'w') as f:
        json.dump(coco_format_annotations, f)



def create_pruned_dataset_common_classes(original_file, input_file, output_file, threshold):
    """
    Generate an alternative pruned dataset based on rarity threshold.
    Logic: Look at classes that have now been undereprented after the first pass of pruned (Based on threshold)
           The classes from these images are then compared with what already exists in the dataset to avoid duplication
           They are then added into the pruned dataset

    Args:
        original_file (str): Path to the original COCO annotations file.
        input_file (str): Path to the input pruned dataset file.
        output_file (str): Path to save the alternative pruned dataset in COCO format.
        threshold (int): Threshold for categorizing common and rare classes.

    Returns:
        None
    """
    # Load pruned dataset and get rare classes
    coco = COCO(original_file)
    coco_pruned = COCO(input_file)
    class_occurrences = {}

    #Original_Block - Get common and rare classes
    common_classes = []
    rare_classes = []

    for cat_id, category in coco.cats.items():
        if len(coco.getImgIds(catIds=[cat_id])) > threshold:
            common_classes.append(cat_id)
        else:
            rare_classes.append(cat_id)

    #Pruned_Block
    for ann in coco_pruned.dataset['annotations']:
        category_id = ann['category_id']

        if category_id not in class_occurrences:
            class_occurrences[category_id] = 1
        else:
            class_occurrences[category_id] += 1

    rare_classes_pruned = [class_id for class_id, occurrences in class_occurrences.items() if occurrences < threshold]

    rare_set = set(rare_classes_pruned) - set(rare_classes)
    rare_classes_pruned_but_common_coco = rare_set.intersection(set(common_classes))
    rare_classes_pruned_but_common_coco = list(rare_classes_pruned_but_common_coco)

    # Load original COCO dataset
    included_classes = rare_classes_pruned_but_common_coco
    pruned_dataset_alt = []

    # Ensure common image ids go first and then rare image ids
    image_ids_with_rare_classes = []

    for img_id in coco.getImgIds():
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        contains_rare_class = any(ann['category_id'] in rare_classes for ann in anns)

        if contains_rare_class:
            image_ids_with_rare_classes.append(img_id)

    image_ids_without_rare_classes = [img_id for img_id in coco.getImgIds() if img_id not in image_ids_with_rare_classes]

    image_ids = image_ids_without_rare_classes + image_ids_with_rare_classes

    annotations = []

    for img_id in image_ids:
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = coco.loadAnns(ann_ids)

        contains_included_classes = any(ann['category_id'] in included_classes for ann in anns)
        contains_other_classes = any(ann['category_id'] not in included_classes for ann in anns)

        if contains_included_classes and not contains_other_classes:
            pruned_dataset_alt.append(img_info)

            for ann in anns:
                category_id = ann['category_id']
                bbox = ann['bbox']
                area = ann['area']
                iscrowd = ann['iscrowd']
                annotation_id = ann['id']

                annotations.append({
                    'image_id': img_id,
                    'category_id': category_id,
                    'bbox': bbox,
                    'area': area,
                    'iscrowd': iscrowd,
                    'id': annotation_id
                })

    # Converting to COCO format
    coco_format_annotations = {
        "images": pruned_dataset_alt,
        "annotations": annotations,
        "categories": coco.dataset["categories"]
    }

    with open(output_file, 'w') as f:
        json.dump(coco_format_annotations, f)



def combine_datasets(original_file, input_file_1, input_file_2, output_file):
    """
    Combine two pruned datasets into one.

    Args:
        original_file (str): Path to the original file from where we take categories.
        input_file_1 (str): Path to the first input pruned dataset file.
        input_file_2 (str): Path to the second input pruned dataset file.
        output_file (str): Path to save the combined pruned dataset.

    Returns:
        None
    """
    #Load original_dataset.json
    coco = COCO(original_file)

    # Load pruned_dataset.json
    with open(input_file_1, 'r') as f:
        pruned_dataset_1 = json.load(f)

    # Load pruned_dataset_alt.json
    with open(input_file_2, 'r') as f:
        pruned_dataset_2 = json.load(f)

    # Combine images and annotations
    combined_images = pruned_dataset_1['images'] + pruned_dataset_2['images']
    combined_annotations = pruned_dataset_1['annotations'] + pruned_dataset_2['annotations']

    # Create a dictionary for the combined dataset
    combined_dataset = {
        'images': combined_images,
        'annotations': combined_annotations,
        'categories': coco.dataset["categories"]  
    }

    # Save the combined dataset as a JSON file
    with open(output_file, 'w') as f:
        json.dump(combined_dataset, f)



def augment_dataset(source_images, input_json, output_json):
    """
    Augment a COCO-style dataset with random transformations and save the augmented dataset to a JSON file.

    Args:
        src_images (str): Path to the source folder containing the images. (Ex: 'val2017')
        input_json (str): Path to the input COCO-style JSON file.
        output_json (str): Path to save the augmented COCO-style JSON file.

    Returns:
        None

    This function performs the following augmentations on the input dataset:
    - Cropping: Randomly crops images to a size between 20% and 80% of the original size with 50% probability.
    - VerticalFlip: Randomly flips images vertically with a 50% probability.
    - HorizontalFlip: Randomly flips images horizontally with a 50% probability.
    - Rotation: Randomly rotates images up to 45 degrees with a 50% probability.

    The bounding box annotations are updated based on the applied augmentations, and the area, iscrowd, and id fields are preserved in the updated annotations.
    """
    # Load pruned_dataset.json
    with open(input_json, 'r') as f:
        pruned_dataset = json.load(f)

    images = pruned_dataset['images']
    annotations = pruned_dataset['annotations']

    # Define a crop transformation
    bbox_params = BboxParams(format='coco', label_fields=['category_id'])

    # Define a list to store updated annotations
    updated_annotations = []

    for img_info in images:
        image_path = os.path.join(source_images, img_info['file_name'])  # Assuming images are in 'val2017' folder
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get annotations for this image
        image_annotations = [ann for ann in annotations if ann['image_id'] == img_info['id']]

        # Define cropping coordinates
        x_min = int(0.2 * image.shape[1])  # 20% of image width
        y_min = int(0.2 * image.shape[0])  # 20% of image height
        x_max = int(0.8 * image.shape[1])  # 80% of image width
        y_max = int(0.8 * image.shape[0])  # 80% of image height

        transform = Compose([
            OneOf([
                Crop(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, p=0.5),
                VerticalFlip(p=0.5),
                HorizontalFlip(p=0.5),
                Rotate(p=0.5, limit=45)  # Rotate up to 45 degrees
            ], p=1),
        ], bbox_params=bbox_params, p=1)

        # Apply augmentation
        augmented = transform(image=image, bboxes=[ann['bbox'] for ann in image_annotations], category_id=[ann['category_id'] for ann in image_annotations])

        # Updated annotations after crop
        for ann in image_annotations:
            category_id = ann['category_id']
            bbox = ann['bbox']

            # Calculate area manually
            x_min, y_min, width, height = bbox
            area = width * height

            updated_annotations.append({
                'image_id': img_info['id'],
                'category_id': category_id,
                'bbox': bbox,
                'area': area,
                'iscrowd': ann['iscrowd'],
                'id': ann['id']
            })

    # Create pruned_dataset_cropped.json structure
    pruned_dataset_cropped = {
        'images': images,
        'annotations': updated_annotations,
        'categories': pruned_dataset['categories']
    }

    # Save pruned_dataset_cropped.json
    with open(output_json, 'w') as f:
        json.dump(pruned_dataset_cropped, f)


