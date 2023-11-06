import numpy as np
import pandas as pd
import argparse
import json
from pycocotools.coco import COCO
import os

def add_additional_classes(original_file, input_file, output_file): 
    '''
    Bring in 2 files for concatenating the json annotations
    new_coco_json is the new annotations to be updated for appropriate label values and then added to the original json for a new combined file
    
    Args: 
        original_file (str): original COCO annotations file (or starting point file if additional classes have already been added to this file)
        input_file (str): JSON file with annotations for new classes to be updated appropriately (start labels at max category value of original)
        output_file (str): string with file location and name of new concatenated annotations file
    
    Returns: 
        none
    '''
    
    # Bring in new file
    ann_file = new_coco_json
    coco=COCO(ann_file)
    
    # Bring in original file
    coco_ann_file = og_coco_json
    coco_real=COCO(coco_ann_file)
    
    # Find maximum label in original file
    max_cat_label = max([cat['id'] for cat in coco_real.dataset['categories']])
    
    # Number of new categories to be added
    num_new_classes = len([cat['id'] for cat in coco.dataset['categories']])
    
    # Update new categories to have labels above original maximum value
    categories_change = coco.dataset['categories'].copy()
    for cat in categories_change: 
        cat['id'] += max_cat_label
        if cat['id'] == max_cat_label: 
            cat['id'] = max_cat_label + num_new_classes
         
    # change annotations to match category label changes
    ann_change = coco.dataset['annotations'].copy()
    for ann in ann_change:
        ann['category_id'] += max_cat_label
        if ann['category_id'] == max_cat_label: 
            ann['category_id'] = max_cat_label + num_new_classes
    
    # Concatenate datasets (add doors to end of COCO)
    coco_ann = coco_real.dataset['annotations'].copy()
    coco_cat = coco_real.dataset['categories'].copy()
    coco_images = coco_real.dataset['images'].copy()

    coco_ann.extend(ann_change)
    coco_cat.extend(categories_change)
    coco_images.extend(coco.dataset['images'].copy())

    coco_format_annotations = {
        "images": coco_images,
        "annotations": coco_ann,
        "categories": coco_cat
    }

    with open(output_file, 'w') as f:
        json.dump(coco_format_annotations, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--add_additional_classes', action='store_true', help='Create dataset with additonal file')

    parser.add_argument('--input_source_file', type=str, help='Path to input source file')
    parser.add_argument('--input_source_file_2', type=str, help='Path to second input source file')
    parser.add_argument('--output_source_file', type=str, help='Path to output source file')

    args = parser.parse_args()

    if args.add_additional_classes and args.input_source_file and input_source_file_2 and args.output_source_file:
        add_additional_classes(args.input_source_file, args.input_source_file_2, args.output_source_file)