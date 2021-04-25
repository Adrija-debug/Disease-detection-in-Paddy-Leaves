# Disease-detection-in-Paddy-Leaves
This project aims to identify three different diseases, namely bacterial leaf blight, brown spot and leaf smut along with identifying healthy leaves in paddy plant using the combination of basic image processing and ML algorithm.
The code is written in MATLAB and SVM alogorithm is used for classification.
The steps involved are as follows:
1. First we put the dataset for the healthy leaves together in one folder. Here, the images from 1 to 305 are for bacterial blight, 306 to 608 are for leaf smut and the remaining are for brown spot. The dataset for the healthy leaves are stored in a separate folder.
2. Next we run the file TrainingAndAccuracyDatasets.m .This creates the matrices having all labels for the particular disease/healthy leaves and the various features extracted.
3. Lastly, run DIPprojectPlantDiseaseDetection.m
