# COCO-Balancer

The project focuses on taking COCO (Common Objects in Context), a popular Computer Vision dataset, and trying to balance it. This is achieved by eliminating images based on our classification of a rare class and a common class.

Run the following after reading run_all:
```
./run_all.sh
```

## Usage

### Creating a pruned dataset:
```
python main.py --create_pruned_dataset --input_source_file instances_val2017.json --output_source_file val_pruned.json --threshold 100
```

### Augmenting your dataset (when you want to retain its size):
```
python main.py --create_augmented_dataset --input_source_file val_pruned.json --output_source_file val_pruned_augmented.json
```

### Augmenting your dataset (when you want to double it in its size):
```
python main.py --create_augmented_dataset_doubled --input_source_file val_pruned.json --output_source_file val_pruned_augmented.json
```

### Adding additional classes to your COCO file:
```
python add_classes.py --add_additional_classes --input_source_file instances_val2017.json --input_source_file_2 doors.json --output_source_file instances_val2017_with_doors.json
```

### Converting COCO to yolo format for modeling:
```
python modeling.py --convert_coco_to_yolo --input_source_file val_pruned_augmented_with_doors.json --output_dir yolov8_dataset/labels
```


### Additional functions to validate your results (helper.py)
Counts files in a folder (Use this to validate if your dataset has been downloaded properly):
```
python helper.py --count_files_in_directory --images_dir val2017
```

Create a pie chart for class counts and output total images (This can be used to visualize your class occurrences so one can do an eye test on how well the dataset is balanced):
```
python helper.py --visualize_class_occurrences --input_source_file val_pruned_augmented.json
```

Create a folder of images with bounding boxes (This helps you visualize the images with annotations up to a certain image number):
```
python helper.py --visualize_images_with_bbox --images_dir val2017 --input_source_file val_pruned_augmented.json --threshold 10
```

Calculate the Shannon entropy of your dataset (This is used as a metric to calculate "balance"):
```
python helper.py --shannon_entropy --input_source_file val_pruned_augmented.json
```

Outputs if two JSONs are similar or not (This function helps validate if augmentations/pruning worked properly):
```
python helper.py --compare_json  --input_source_file val_pruned_augmented.json --input_source_file_2 val_pruned.json
```


## Results from val2017
This folder has been added to visualize and also provide the files inputted/outputted through this code.

*Results*: The Shannon Diversity Index on COCO (for validation set): 0.78. The Shannon Diversity Index on COCO_Balanced (for validation set): 0.85
