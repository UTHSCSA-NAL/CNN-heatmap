# CNN Heatmaps Can Capture Brain Alterations Induced by Alzheimer's Disease
This code implements the training of 3D convolutional neural networks and generating heatmaps, as described in the paper CNN Heatmaps Can Capture Brain Alterations Induced by Alzheimer's Disease.

**Di Wang, Nicolas Honnorat, Mohamad Habes**

**Abstract:**
Deep networks are reaching excellent performances for clinical tasks such as the generation of individual diagnosis and prognosis from MRI images. But the complexity of these models make them difficult to interpret. Several heatmaps methods were developed to address this issue by highlighting the images features that are used by the deep networks to support their decision, but in many fields of research, the lack of ground-truth limited the possibilities of estimating the quality of the heatmaps generated. In this work, we explain how this issue can be address, in the neuroimaging field, by comparing heatmaps with brain maps derived from meta-analysis studies. More specifically, we compare a meta-analysis map derived by combining 79 voxel-based morphometry studies with the heatmaps generated, for a CNN reaching a good accuracy during the classification of ADNI participant with and without Alzheimer's Disease, by three prominent methods: layer-wise relevance propagation (LRP), integrated gradients (IG) and guided grad-CAM (GGC). Our results indicate that the three heatmap methods capture brain regions overlapping the meta-analysis map, and more relevant than maps derived from Support Vector Machines. GGC produced the best overlaps.

Please cite the last version of the article when it will be published.

version 1.0.0
author: Di Wang
date: March 28, 2022

# Usage
The usage of the python scripts can be printed by executed them with the -h option.

# Example
Example of training model with five fold cross validation and early-stoping: 
```
python3 train_cv.py --path ./data.csv --outpath ./output/ --device 0 --model modelA24
```

Example of generating heatmaps from trained model: 
```
python3 heatmap.py --path ./data.csv --modelpath ./output/ --model modelA24 --method LRP --outpath ./LRP/ --device 0 
python3 heatmap.py --path ./data.csv --modelpath ./output/ --model modelA24 --method IG --outpath ./LRP/ --device 0 
python3 heatmap.py --path ./data.csv --modelpath ./output/ --model modelA24 --method GGC --outpath ./LRP/ --device 0 
```
