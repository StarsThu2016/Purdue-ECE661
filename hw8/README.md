# ECE 661 hw08 -- Camera calibration

## Homework text
hw8.pdf

## My Report
hw8_RanXu.pdf

## Source Code (Requiring Jupyter Notebook to Open and Run)
$ jupyter notebook

Go through all cells in CamCal.ipynb. Reulsts will be saved in CannyResults1/, CannyResults2/, HoughLines1/, HoughLines2/, HoughPoints1/, HoughPoints2/, ReProjLinear1/, ReProjLinear2/, ReProjLM1, and ReProjLM2/

## To Install Jupyter Notebook in Ubuntu 16.04
python3 -m pip install --upgrade pip

python3 -m pip install jupyter

## Required Python3 Packages
numpy, matplotlib, cv2

## Result in first dataset -- 40 given images (Files/Dataset1/)
The original images are in 480H x 640W x 3 shape.

The reprojection error is as follows, 

| Method                       | Mean Eur. Dis. | Var. Eur. Dis. | 
| ---------------------------- | :------------: | :------------: | 
| w/o LM                       |       2.17     |      2.55      | 
| w/ LM, w/o radial distortion |       0.88     |      0.28      | 

The summed square projection error in LM is as follows, 

| Method                       |      Cost      | 
| ---------------------------- | :------------: | 
| w/o LM                       |    4419.28     | 
| w/ LM, w/o radial distortion |     840.47     | 
| w/ LM, w/ radial distortion  |     709.15     | 

Intrinsic parameter after LM w/o radial distortion
K =  [[720.1,   1.93, 322.81],
      [  0. , 717.55, 245.11],
      [  0. , 0.    , 1.  ]]
 
## Result in second dataset -- 21 own images (Files/Dataset2/)
The original images are in 4032H x 3024W x 3 shape, I downsampled them to 512H x 384W x 3 shape (ratio = 7.875).
The resulting K right multiplies diag([7.875, 7.875, 1]) equals the original K, while R and t keeps the same.

The error in the reshaped domain is as follows, 

| Method                       | Mean Eur. Dis. | Var. Eur. Dis. | 
| ---------------------------- | :------------: | :------------: | 
| w/o LM                       |       1.42     |      1.90      | 
| w/ LM, w/o radial distortion |       0.98     |      0.75      | 

The summed square projection error in LM is as follows, 

| Method                       |      Cost      | 
| ---------------------------- | :------------: | 
| w/o LM                       |    1348.88     | 
| w/ LM, w/o radial distortion |     689.72     | 
| w/ LM, w/ radial distortion  |     576.01     | 

Intrinsic parameter after LM w/o radial distortion
K =  [[418.0, 0.048, 196],
      [  0. , 416.0, 258],
      [  0. , 0.    , 1.]]

Note that the mean error is 8x, var error and cost is 64x in the original shape domain. The only way to reduce 
the error in the original shape domain is to represent the point correspondences in the original shape.
