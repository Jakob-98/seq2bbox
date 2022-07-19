

## Step by step preprocessing for thesis research reproducibility:
1. Download the ENA dataset and channel islands datasets from https://lila.science/datasets/channel-islands-camera-traps/ and https://lila.science/datasets/ena24detection. I suggest using `azcopy` as the datasets are quite large (89GB total). Take note of the paths to the images as well as the paths to the metadata (json) files. 
2. Install the dependencies in requirements.txt
3. Run the dataprep python files found in /datasethandler/*.py to generate the pickle files for the data subsets
4. Using these pickle files, the single-image or sequential image datapipeline can be run. Edit the `config` class in the `datapipeline.py` file and run it. 




### References: 

Letterbox: https://github.com/ultralytics/yolov5/blob/4d157f578a7bbff08d1e17a4e6e47aece4d91207/utils/augmentations.py#L91

Local Binary Patterns (mahotas): https://mahotas.readthedocs.io/en/latest/index.html

Part of background substraction logic: https://stackoverflow.com/questions/60646384/python-opencv-background-subtraction-and-bounding-box


seaborn plots: https://www.statology.org/seaborn-barplot-show-values/

TODO: createlabels (bbox label) coco to yolo...

