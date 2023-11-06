#!/bin/bash

#Have the following files in your working directory:
#instances_train2017.json
#additional_class_train.json
#instances_val2017.json
#additional_class_val.json

#Train
#Command 1
python add_classes.py --add_additional_classes --input_source_file instances_train2017.json --input_source_file_2 additional_class_train.json --output_source_file instances_train2017_with_additional_class.json

# Command 2
python main.py --create_pruned_dataset --input_source_file instances_train2017_with_additional_class.json --output_source_file instances_train2017_with_additional_class_pruned.json --threshold 2500

# Command 3
python main.py --create_augmented_dataset --input_source_file instances_train2017_with_additional_class_pruned.json --output_source_file instances_train2017_with_additional_class_pruned_augmented.json

# Command 4
python modeling.py --convert_coco_to_yolo --input_source_file instances_train2017_with_additional_class_pruned_augmented.json --output_dir yolo_dataset_train/labels


#Validation
#Command 1
python add_classes.py --add_additional_classes --input_source_file instances_val2017.json --input_source_file_2 additional_class_val.json --output_source_file instances_val2017_with_additional_class.json

# Command 2
python main.py --create_pruned_dataset --input_source_file instances_val2017_with_additional_class.json --output_source_file instances_val2017_with_additional_class_pruned.json --threshold 100

# Command 3
python main.py --create_augmented_dataset --input_source_file instances_val2017_with_additional_class_pruned.json --output_source_file instances_val2017_with_additional_class_pruned_augmented.json

# Command 4
python modeling.py --convert_coco_to_yolo --input_source_file instances_val2017_with_additional_class_pruned_augmented.json --output_dir yolo_dataset_val/labels
