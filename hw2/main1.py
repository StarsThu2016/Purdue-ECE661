# This is the main python script for ECE 661 hw2
# To run: python3 main.py
import numpy as np
from PIL import Image
from homography_util import *

A = np.array([[1467,75], [2985,681], [1440,2340], [3048,2094]])  
B = np.array([[1278,258], [3054,573], [1245,2082], [3081,1938]])
C = np.array([[876,690], [2866,300], [856,2142], [2922,2312]])
D = np.array([[0,0], [1279,0], [0,719], [1279,719]])
imgA = np.array(Image.open("PicsHw2/1.jpg"))
imgB = np.array(Image.open("PicsHw2/2.jpg"))
imgC = np.array(Image.open("PicsHw2/3.jpg"))
imgD = np.array(Image.open("PicsHw2/Jackie.jpg"))

# Task 1a
# Hymnographies and resulting images from Fig. 1(d) to Fig. 1(a)
H_DA = Homography(D,A)
print("H_DA = ", H_DA)
print(np.sum(imgA))
RefilledA_WithD = Filling(imgA, A, imgD, D)
print(np.sum(imgA))
RefilledImgA_WithD = Image.fromarray(RefilledA_WithD, 'RGB')
RefilledImgA_WithD.save("Task1a1.jpg")

# Homographies and resulting images from Fig. 1(d) to Fig. 1(b)
H_DB = Homography(D,B)
print("H_DB = ", H_DB)
RefilledB_WithD = Filling(imgB, B, imgD, D)
RefilledImgB_WithD = Image.fromarray(RefilledB_WithD, 'RGB')
RefilledImgB_WithD.save("Task1a2.jpg")

# Homographies and resulting images from Fig. 1(d) to Fig. 1(c)
H_DC = Homography(D,C)
print("H_DC = ", H_DC)
RefilledC_WithD = Filling(imgC, C, imgD, D)
RefilledImgC_WithD = Image.fromarray(RefilledC_WithD, 'RGB')
RefilledImgC_WithD.save("Task1a3.jpg")
print("----------")

# Task 1b
# Homographies from Fig. 1(a) to Fig. 1(b)
H_AB = Homography(A,B)
print("H_AB = ", H_AB)

# Homographies from Fig. 1(b) to Fig. 1(c)
H_BC = Homography(B,C)
print("H_BC = ", H_BC)

# Homographies from Fig. 1(a) to Fig. 1(c)
H_AC = Homography(A,C)
print("H_AC = ", H_AC)

H_ABC = np.matmul(H_BC, H_AB)
H_ABC = H_ABC/H_ABC[2][2]
print("H_ABC = ", H_ABC)

# Mapping the imgA to imgC plane
(x_min, y_min, mappedA) = Mapped(imgA, H_AC)
print("(x_min, y_min) = (%d, %d)" %(x_min, y_min))
mappedimgA = Image.fromarray(np.uint8(mappedA), 'RGB')
mappedimgA.save("Task1b.jpg")
