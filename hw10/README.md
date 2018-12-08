# ECE 661 hw10 -- Face Recognition and Object Detection

## Homework text
hw10.pdf

## My Report
hw10_RanXu.pdf

# Face recognition task
Face images are all in (128,128,3) shape. We need to train on 30 classes x 21 images per classes and test on another 21 images per class. The stats are shown as follows,

| Item                         | Train Set | Test Set | 
| ---------------------------- | :-------: | :------: | 
| Number of people             |     30    |    30    | 
| Number of faces per people   |     21    |    21    | 

## Precision in test set
Both methods achieve 100% test accuracy when 9 eigenvectors are preserved. LDA is always better than PCA.

# Car detection dataset
Car images are all in (20, 40, 3) shape. We need to train on 710 positive samples and 1758 negative samples. We want to test on 178 positive samples and 440 negative samples, the stats are shown as follows,

| Item                         | Train Set | Test Set | 
| ---------------------------- | :-------: | :------: | 
| Positive                     |    710    |   178    | 
| Negative                     |   1758    |   440    | 

Use "'ECE661_2018_hw10_DB2/%s/%s/%06d.png' %(TrainOrTest, PosOrNeg, x) for x range(1, N+1)" to index them. 

## Precision in test set
I have two training runs. The first one splits the training in first stage and later stage and has some weakness. The second one finishes the training in one scripts and adds an early termination condition. The precision of the first run, a.k.a. the one shown in the report, is as follows,

| Detection\Ground Truth       |  True     |  False   |   Sum  |
| ---------------------------- | :-------: | :------: | :----: | 
| Positive                     |   136     |    8     |        | 
| Negative                     |    42     |  432     |        |
| Sum                          |   178     |  440     |   618  |

False positive rate (must be low): FP/(FP+TN) = 1.82%

False negative rate (higher the better): FN/(TP+FN) = 23.60% 

The precision of the second run is as follows,

| Detection\Ground Truth       |  True     |  False   |   Sum  |
| ---------------------------- | :-------: | :------: | :----: | 
| Positive                     |   129     |    5     |        | 
| Negative                     |    49     |  435     |        |
| Sum                          |   178     |  440     |   618  |

False positive rate (must be low): FP/(FP+TN) = 1.14%

False negative rate (higher the better): FN/(TP+FN) = 27.53%

## Source Code (Requiring Jupyter Notebook to Open and Run)
$ jupyter notebook

Go through all cells in FaceRecog.ipynb for face recognition task.

Due to the copyright issue, I cannot release the dataset. So, only persons who get access to the second dataset can re-run my scripts on car detection task. 

Go through all cells in CarDetection1_FirstEpoch.ipynb, CarDetection1_NextEpoch.ipynb and CarDetectionVerifyAcc1.ipynb (first run), CarDetection2_Training.ipynb and CarDetectionVerifyAcc2.ipynb (second run).

To have non-interactive server training, use the following commands for the first run in the car detection task,

python3 CarDetection1_FirstEpoch.py 10 5 200 10

python3 CarDetection1_NextEpoch.py 9 20 2 2

To have non-interactive server training, use the following commands for the second run in the car detection task,

python3 CarDetection2_Training.py 10 20 2 10 0.9 0.3

## To Install Jupyter Notebook in Ubuntu 16.04
python3 -m pip install --upgrade pip

python3 -m pip install jupyter

## Required Python3 Packages
numpy, matplotlib, cv2, pickle
