# Deep neural network heatmaps capture Alzheimer's disease patterns reported in a large meta-analysis of neuroimaging studies
This code implements the training of 3D convolutional neural networks and generating heatmaps, as described in the paper CNN Heatmaps Can Capture Brain Alterations Induced by Alzheimer's Disease.

**Di Wang, Nicolas Honnorat, Mohamad Habes**

**Abstract:**
Deep neural networks currently provide the most advanced and accurate machine learning models to distinguish between structural MRI scans of subjects with Alzheimer's disease and healthy controls. Unfortunately, the subtle brain alterations captured by these models are difficult to interpret because of the complexity of these multi-layer and non-linear models. Several heatmap methods have been proposed to address this issue and analyze the imaging patterns extracted from the deep neural networks, but no quantitative comparison between these methods has been carried out so far. In this work, we explore these questions by deriving heatmaps from Convolutional Neural Networks (CNN) trained using T1 MRI scans of the ADNI data set and by comparing these heatmaps with brain maps corresponding to Support Vector Machines (SVM) coefficients. Three prominent heatmap methods are studied: Layer-wise Relevance Propagation (LRP), Integrated Gradients (IG), and Guided Grad-CAM (GGC). Contrary to prior studies where the quality of heatmaps was visually or qualitatively assessed, we obtained precise quantitative measures by computing overlap with a ground-truth map from a large meta-analysis that combined 77 voxel-based morphometry (VBM) studies independently from ADNI. Our results indicate that all three heatmap methods were able to capture brain regions covering the meta-analysis map and achieved better results than SVM coefficients. Among them, IG produced the heatmaps with the best overlap with the independent meta-analysis.

Please cite the last version of the article when it will be published.

version 1.0.0
author: Di Wang
date: July 26, 2022

# Usage
The usage of the python scripts can be printed by executed them with the -h option.
```
python3 train_cv.py -h
```

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
